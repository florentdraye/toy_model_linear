import unittest
import torch

from remeasure_emergence_best_site import all_depth_means, choose_cells, calibration_scores
from src.config import ModelConfig
from src.model import ToyTransformer
from src.uniform_means import UniformMeanBank, hidden_at
from src.emergence import measure


class BestSiteTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(27)
        self.model = ToyTransformer(ModelConfig(vocab_size=3, seq_len=3, n_classes=3,
            d_model=8, n_heads=2, d_ff=16, n_blocks=2)).eval()

    def test_all_depth_means_match_separate_forwards(self):
        edges = torch.tensor([[0, 1, 2], [2, 1, 0], [1, 1, 2], [2, 2, 2], [1, 0, 1]])
        bank = UniformMeanBank(edges, torch.tensor([10, 20, 20, 30, 30]))
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        means = all_depth_means(self.model, bank, batch_size=2)
        for depth in (0, 1, 2):
            _, expected = bank.fit(self.model, bank.classes, depth, [0, 1, 2], batch_size=2)
            torch.testing.assert_close(means[depth], expected, atol=1e-7, rtol=1e-6)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.model.state_dict().items()))

    def test_selection_is_per_latent_and_uses_raw_calibration(self):
        cube = torch.full((2, 3, 2, 2), -.9)
        cube[1, 0, 0, 0] = -.2
        cube[0, 2, 1, 1] = .8
        choices = choose_cells(cube, [1., 4.])
        self.assertEqual([(c['depth'], c['position'], c['alpha']) for c in choices],
                         [(1, 0, 1.), (0, 2, 4.)])
        unit = choose_cells(cube, [1., 4.], unit_only=True)
        self.assertEqual((unit[1]['depth'], unit[1]['position'], unit[1]['alpha']), (0, 0, 1.))
        self.assertLess(choices[0]['calibration_skill'], 0.)

    def test_calibration_scoring_matches_hook_metric(self):
        off = torch.tensor([[[0, 1, 2], [2, 1, 0]]])
        on = torch.tensor([[[1, 1, 2], [2, 2, 0]]])
        labels = torch.tensor([[0, 1]])
        vector = torch.randn(1, 3, 8) * .1
        for depth in (0, 1, 2):
            hidden = hidden_at(self.model, off[0], depth)[None]
            score = calibration_scores(self.model, hidden, labels, vector, depth, 2, 4.)
            expected = measure(self.model, dict(off=off, on=on, y_on=labels, y_off=1-labels),
                               4*vector[:, 2:3], depth, [2])
            self.assertAlmostEqual(float(score[0]), expected['steer_raw'][0], places=6)


if __name__ == '__main__':
    unittest.main()
