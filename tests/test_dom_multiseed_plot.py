import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from plot_dom_multiseed import load_runs


class AggregationTest(unittest.TestCase):
    def test_common_grid_preserves_negative_scores_and_equal_seed_weight(self):
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for seed, steps, score in [(46, [0, 100, 200], -0.8), (47, [0, 100], 1.)]:
                run = dict(config=dict(seed=seed, eval_every=100, steps=200, best_locations=True,
                                       graph_seed=0, data_seed=314), complete=False,
                           latents=[2], frequency=[.1], reference=0, model_config={},
                           evaluation_bank_sha256='fixed', training_support_sha256='fixed', history=[])
                for t in steps:
                    row = dict(step=t)
                    for key in ('gain_raw', 'steer_raw', 'patch_raw', 'accuracy', 'steering_accuracy', 'patch_accuracy'):
                        row[key] = [score]
                    run['history'].append(row)
                path = Path(root)/f'{seed}.json'; path.write_text(json.dumps(run)); paths.append(path)
            _, steps, _, arrays = load_runs(paths, allow_partial=True)
            np.testing.assert_array_equal(steps, [0, 100])
            np.testing.assert_allclose(arrays['steer_raw'].mean(0), .1)
            with self.assertRaises(ValueError): load_runs(paths)
            with self.assertRaises(ValueError): load_runs([paths[0], paths[0]], True)


if __name__ == '__main__':
    unittest.main()
