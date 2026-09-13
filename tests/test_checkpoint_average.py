import unittest
import torch
from src.checkpoint_average import CheckpointAverage


class CheckpointAverageTest(unittest.TestCase):
    def test_causal_recurrence_and_ownership(self):
        avg = CheckpointAverage(.75)
        first = {'w': torch.tensor([2., 4.]), 'count': torch.tensor(1)}
        avg.update(0, first)
        first['w'].zero_()
        avg.update(200, {'w': torch.tensor([6., 8.]), 'count': torch.tensor(2)})
        torch.testing.assert_close(avg.state['w'], torch.tensor([3., 5.]))
        before = avg.state['w'].clone()
        avg.update(400, {'w': torch.tensor([7., 9.]), 'count': torch.tensor(3)})
        torch.testing.assert_close(before, torch.tensor([3., 5.]))
        torch.testing.assert_close(avg.state['w'], torch.tensor([4., 6.]))
        self.assertEqual(int(avg.state['count']), 3)
        with self.assertRaises(ValueError):
            avg.update(200, first)

    def test_zero_decay_reproduces_snapshot(self):
        avg = CheckpointAverage(0)
        avg.update(0, {'w': torch.tensor([1.])})
        avg.update(200, {'w': torch.tensor([5.])})
        torch.testing.assert_close(avg.state['w'], torch.tensor([5.]), rtol=0, atol=0)
