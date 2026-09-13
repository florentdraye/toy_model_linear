"""Exact class balancing, per-position means, split isolation, and last-token edit."""
import unittest
import torch

from src.config import GraphConfig, ModelConfig
from src.data import enumerate_paths, split_indices
from src.emergence import forward_at
from src.graph import Graph
from src.model import ToyTransformer
from src.uniform_means import UniformMeanBank, hidden_at
from train_emergence import parser
from scan_emergence_depths import calibration_ids


class UniformMeanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setup_model(self):
        torch.manual_seed(19)
        return ToyTransformer(ModelConfig(vocab_size=3, seq_len=3, n_classes=3,
                                           d_model=8, n_heads=2, d_ff=16, n_blocks=2)).eval()

    def test_full_training_support_and_exact_uniform_negative(self):
        model = self.setup_model()
        paths = enumerate_paths(Graph(GraphConfig(4, (1, 3, 3, 3), 3, 2)))
        train, test = split_indices(len(paths['labels']), .8, 71)
        edges, labels = paths['edge_seqs'][train], paths['nodes'][train, 2]
        bank = UniformMeanBank(edges, labels)
        # Exactly one contribution per training trajectory; held-out paths absent.
        def ids(e):
            return (e * torch.tensor([9, 3, 1])).sum(-1)
        self.assertEqual(set(ids(bank.edges).tolist()), set(train.tolist()))
        self.assertFalse(set(ids(bank.edges).tolist()) & set(test.tolist()))
        targets, positions = bank.classes, [0, 1, 2]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        v, means = bank.fit(model, targets, 1, positions, batch_size=2)
        _, h = forward_at(model, edges, 1, positions, capture=True)
        direct = torch.stack([h[labels == k].double().mean(0) for k in targets])
        self.assertTrue(torch.allclose(means, direct, atol=1e-7, rtol=1e-6))
        for j, target in enumerate(targets):
            negative = direct[torch.tensor(targets) != target].mean(0)
            self.assertTrue(torch.allclose(v[j].double(), direct[j]-negative, atol=1e-7, rtol=1e-6))
        v_last, _ = bank.fit(model, targets, 1, [2], batch_size=100)
        self.assertTrue(torch.allclose(v_last[:, 0], v[:, 2], atol=1e-7, rtol=1e-6))
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(t, before[k]) for k, t in model.state_dict().items()))
        self.assertFalse(v.requires_grad)
        info = bank.description(targets)
        self.assertEqual(info['total_paths'], len(train))
        self.assertTrue(all(p+n == len(train) for p, n in zip(info['positive_paths'], info['negative_paths'])))

    def test_negative_classes_have_equal_weight_despite_path_counts(self):
        model = self.setup_model()
        edges = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 1, 0], [2, 0, 1]])
        labels = torch.tensor([10, 20, 30, 30, 30])
        bank = UniformMeanBank(edges, labels)
        v, means = bank.fit(model, [10], 1, [2], 2)
        self.assertTrue(torch.allclose(v[0].double(), means[0]-(means[1]+means[2])/2, atol=1e-7))
        pooled = means[0]-(means[1]+3*means[2])/4
        self.assertGreater(float((v[0]-pooled).norm()), 1e-5)

    def test_only_last_token_is_edited_and_prefix_forward_matches(self):
        model = self.setup_model()
        edges = torch.tensor([[0, 1, 2], [1, 0, 2]])
        h = hidden_at(model, edges, 1)
        _, captured = forward_at(model, edges, 1, [0, 1, 2], capture=True)
        self.assertTrue(torch.equal(h, captured))
        seen = []
        handle = model.blocks[1].register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach()))
        delta = torch.arange(8).float()[None] / 100
        try:
            forward_at(model, edges, 1, [2], delta)
        finally:
            handle.remove()
        self.assertTrue(torch.equal(seen[0][:, :2], h[:, :2]))
        self.assertTrue(torch.allclose(seen[0][:, 2], h[:, 2] + delta))
        self.assertEqual(len(model.blocks[0]._forward_hooks), 0)

    def test_defaults_and_invalid_support(self):
        args = parser().parse_args(['--out-dir', 'unused'])
        self.assertEqual(args.site, 'last')
        self.assertEqual(args.direction_method, 'uniform-mean')
        with self.assertRaises(ValueError):
            UniformMeanBank(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, dtype=torch.long))
        model = self.setup_model()
        bank = UniformMeanBank(torch.zeros(2, 3, dtype=torch.long), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            bank.fit(model, [2], 1, [2])
        model.train()
        with self.assertRaises(ValueError):
            bank.fit(model, [0], 1, [2])

    def test_final_block_patch_reproduces_target_logits(self):
        model = self.setup_model()
        off = torch.tensor([[0, 1, 2], [1, 0, 2]])
        on = torch.tensor([[2, 2, 2], [2, 1, 2]])
        depth = len(model.blocks)
        _, ha = forward_at(model, off, depth, [2], capture=True)
        logits, hb = forward_at(model, on, depth, [2], capture=True)
        patched = forward_at(model, off, depth, [2], delta=hb-ha)
        torch.testing.assert_close(patched, logits, atol=1e-6, rtol=1e-5)

    def test_depth_scan_uses_training_calibration_tail(self):
        saved = dict(train=torch.arange(16), test=torch.tensor([[[16, 17]]]),
                     fit=torch.arange(16).reshape(1, 8, 2))
        train, pairs = calibration_ids(saved, 18)
        self.assertTrue(torch.equal(pairs, saved['fit'][:, 6:]))
        self.assertTrue(torch.equal(train, saved['train']))
        saved['fit'][0, -1, -1] = 17
        with self.assertRaises(ValueError):
            calibration_ids(saved, 18)


if __name__ == '__main__':
    unittest.main()
