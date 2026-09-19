import unittest

import torch

from src.config import ModelConfig
from src.context_transfer import (cosine_rows, encode, encoder_parameters,
                                  evaluation_state, source_gradient,
                                  subtract_step, task_loss, transfer)
from src.model import ToyTransformer


class ContextTransferTest(unittest.TestCase):
    def test_first_order_matches_small_virtual_step(self):
        torch.manual_seed(13)
        model = ToyTransformer(ModelConfig(vocab_size=3, seq_len=3, n_classes=4,
            d_model=4, n_heads=1, d_ff=8, n_blocks=2, mlp_activation="relu2")).double().eval()
        params = encoder_parameters(model)
        source_x = torch.tensor([[0, 1, 2], [1, 2, 0]])
        source_y = torch.tensor([1, 2])
        eval_x = torch.tensor([[2, 0, 1], [1, 0, 2]])
        eval_y = torch.tensor([3, 1])
        direction = source_gradient(model, params, source_x, source_y)
        before, g, _ = evaluation_state(model, params, eval_x, eval_y)
        t = transfer(model, params, eval_x, direction)
        predicted = (g * t).flatten(1).sum(1)
        eta = 1e-5
        after = torch.nn.functional.cross_entropy(
            __import__('src.emergence', fromlist=['from_hidden']).from_hidden(
                model, encode(model, subtract_step(params, direction, eta), eval_x), 1),
            eval_y, reduction='none')
        actual = (before - after) / eta
        torch.testing.assert_close(actual, predicted, rtol=3e-3, atol=2e-5)

    def test_cosine_rows(self):
        a = torch.tensor([[1., 0.], [0., 0.]])
        c, dot, an, bn = cosine_rows(a, a)
        self.assertAlmostEqual(float(c[0]), 1.)
        self.assertTrue(torch.isnan(c[1]))
        self.assertEqual(float(dot[0]), 1.)


if __name__ == '__main__':
    unittest.main()
