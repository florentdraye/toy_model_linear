import unittest
import torch

from src.config import ModelConfig
from src.model import ToyTransformer
from src.uniform_means import UniformMeanBank
from src.dom_locations import all_means, measure_locations
from src.emergence import forward_at, target_fidelity


class LocationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(15)
        self.model = ToyTransformer(ModelConfig(vocab_size=3, seq_len=3, n_classes=3,
                                                d_model=8, n_heads=2, d_ff=16, n_blocks=2)).eval()
        self.edges = torch.randint(3, (20, 3))
        self.bank = UniformMeanBank(self.edges, torch.arange(20) % 4)
        self.pairs = dict(off=self.edges[:8].reshape(2, 4, 3),
                          on=self.edges[8:16].reshape(2, 4, 3),
                          y_on=torch.tensor([[0, 1, 2, 1], [1, 2, 0, 0]]))

    def test_full_support_means_match_existing_estimator(self):
        means = all_means(self.model, self.bank, 7)
        for depth in range(1, 3):
            _, old = self.bank.fit(self.model, [1, 2], depth, [0, 1, 2], 7, reference=0)
            torch.testing.assert_close(means[depth], old)
        keep = torch.arange(20) % 4 != 3
        subset = UniformMeanBank(self.edges[keep], (torch.arange(20) % 4)[keep])
        torch.testing.assert_close(all_means(self.model, subset, 7), means[:, :3])

    def test_selection_is_test_independent_and_matches_hook_replay(self):
        rng = torch.get_rng_state().clone()
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        row, vectors, _ = measure_locations(self.model, self.bank, [1, 2], 0,
                                            self.pairs, self.pairs, 2, 7, 8)
        for j, cell in enumerate(row['choices']):
            pos, depth = cell['positions'], cell['depth']
            logits = forward_at(self.model, self.pairs['off'][j], depth, pos, vectors[j, pos])
            gain, _ = target_fidelity(logits.softmax(-1), self.pairs['y_on'][j])
            self.assertAlmostEqual(gain, row['steer_raw'][j], places=6)
        changed = {**self.pairs, 'y_on': (self.pairs['y_on']+1) % 3}
        second, _, _ = measure_locations(self.model, self.bank, [1, 2], 0,
                                         self.pairs, changed, 2, 7, 8)
        self.assertEqual(row['choices'], second['choices'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in self.model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
