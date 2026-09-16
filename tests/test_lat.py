import unittest
import torch
from src.lat import principal_axes


class LATTests(unittest.TestCase):
    def test_matches_exact_principal_component(self):
        torch.manual_seed(4)
        x = torch.randn(3, 80, 9, dtype=torch.float64)
        x[:, :, 2] += 3
        axis, info = principal_axes(x, tolerance=1e-10)
        c = x.transpose(-1, -2) @ x / x.shape[1]
        values, eig = torch.linalg.eigh(c)
        self.assertTrue(torch.all((axis * eig[..., -1]).sum(-1).abs() > 1 - 1e-8))
        self.assertLess(info['exact_check_error'], 1e-8)

    def test_constant_contrast_retains_signal_and_zero_contrast(self):
        x = torch.zeros(2, 20, 6, dtype=torch.float64)
        x[0, :, 1] = 4
        axis, info = principal_axes(x, tolerance=1e-10)
        torch.testing.assert_close(axis[0], torch.tensor([0., 1., 0., 0., 0., 0.], dtype=torch.float64))
        torch.testing.assert_close(axis[1], torch.zeros(6, dtype=torch.float64))
        torch.testing.assert_close(info['explained_energy'], torch.tensor([1., 0.], dtype=torch.float64))

    def test_projection_is_sign_invariant(self):
        u = torch.tensor([.6, .8])
        mean = torch.tensor([2., -1.])
        torch.testing.assert_close(u * (mean @ u), -u * (mean @ -u))


if __name__ == '__main__': unittest.main()
