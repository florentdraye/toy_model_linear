"""Scientific contracts: frequency law, split isolation, paired targets, edits."""
import unittest
import torch
from src.config import GraphConfig, ModelConfig
from src.graph import Graph
from src.data import enumerate_paths, split_indices, LatentFrequencySampler
from src.emergence import paired_bank, path_ids, fit_directions, forward_at, fidelity, measure


class EmergenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frequency_corrects_path_multiplicity(self):
        labels = torch.tensor([0] * 5 + [1] * 400 + [2] * 25)
        sampler = LatentFrequencySampler(labels, ratio=16, seed=7)
        idx, cls = sampler.sample(100_000, torch.Generator().manual_seed(88))
        self.assertTrue(torch.equal(labels[idx], sampler.classes[cls]))
        observed = torch.bincount(cls, minlength=3) / len(cls)
        self.assertTrue(torch.allclose(observed, sampler.probabilities, atol=.006))
        self.assertAlmostEqual(float(sampler.probabilities.max()/sampler.probabilities.min()), 16, places=4)

    def setup_bank(self):
        graph = Graph(GraphConfig(5, (1, 4, 4, 4, 4), 4, 2))
        paths = enumerate_paths(graph)
        train, test = split_indices(len(paths['labels']), .6, 8)
        allowed = torch.zeros(len(paths['labels']), dtype=torch.bool)
        allowed[train] = True
        targets = paths['nodes'][:, 2].unique().tolist()
        ref = targets.pop(0)
        bank = paired_bank(paths, ~allowed, 2, targets, ref, 12, 9, 4)
        return graph, paths, allowed, targets, ref, bank

    def test_pairs_and_disjoint_support(self):
        graph, paths, allowed, targets, ref, bank = self.setup_bank()
        self.assertFalse(allowed[bank['ids']].any())
        self.assertTrue(torch.equal(bank['off'][..., 2:], bank['on'][..., 2:]))
        self.assertTrue(torch.equal(path_ids(bank['on'], 4), bank['ids'][..., 1]))
        for j, target in enumerate(targets):
            nodes = graph.traverse(bank['on'][j])
            self.assertTrue((nodes[:, 2] == target).all())
            self.assertTrue((graph.traverse(bank['off'][j])[:, 2] == ref).all())
            self.assertTrue(torch.equal(nodes[:, -1], bank['y_on'][j]))

    def test_optimal_direction_and_hook(self):
        graph, paths, allowed, targets, ref, bank = self.setup_bank()
        model = ToyTransformer(ModelConfig(vocab_size=4, seq_len=4, n_classes=4,
                                           d_model=8, n_heads=2, d_ff=16, n_blocks=2)).eval()
        positions = [1, 2, 3]
        v = fit_directions(model, bank, 1, positions, 2)
        for j in range(len(targets)):
            _, ha = forward_at(model, bank['off'][j], 1, positions, capture=True)
            _, hb = forward_at(model, bank['on'][j], 1, positions, capture=True)
            self.assertTrue(torch.allclose(v[j], (hb-ha).mean(0), atol=1e-7))
            best = ((hb-ha-v[j])**2).sum()
            worse = ((hb-ha-v[j]-.01)**2).sum()
            self.assertLess(float(best), float(worse))
        a = bank['off'][0]
        self.assertTrue(torch.equal(model(a), forward_at(model, a, 1, positions, torch.zeros_like(v[0]))))
        self.assertEqual(len(model.blocks[0]._forward_hooks), 0)
        # Replacing ALL positions exactly must reproduce the counterfactual.
        b = bank['on'][0]
        _, ha = forward_at(model, a, 1, [0, 1, 2, 3], capture=True)
        _, hb = forward_at(model, b, 1, [0, 1, 2, 3], capture=True)
        patched = forward_at(model, a, 1, [0, 1, 2, 3], hb-ha)
        self.assertTrue(torch.allclose(patched, model(b), atol=1e-6))
        metrics = measure(model, bank, v, 1, positions, 2)
        self.assertEqual(len(metrics['gain_raw']), len(targets))

    def test_fidelity_null_perfect_wrong_and_zero_teacher_rows(self):
        target = torch.tensor([[1., -1.], [0., 0.]])
        self.assertEqual(fidelity(torch.zeros_like(target), target)[0], 0)
        self.assertEqual(fidelity(target, target)[0], 1)
        self.assertEqual(fidelity(-target, target)[0], -3)
        leaking = target.clone()
        leaking[1] = torch.tensor([1., -1.])
        self.assertEqual(fidelity(leaking, target)[0], 0)


from src.model import ToyTransformer

if __name__ == '__main__':
    unittest.main()
