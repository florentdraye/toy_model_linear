"""Causal parameter averaging of saved snapshots; never averages scores."""
import torch


class CheckpointAverage:
    def __init__(self, decay):
        if not 0 <= decay < 1:
            raise ValueError('decay must lie in [0, 1)')
        self.decay = decay
        self.state = None
        self.step = -1

    @torch.no_grad()
    def update(self, step, state):
        if step <= self.step:
            raise ValueError('snapshots must be strictly chronological')
        if self.state is None:
            self.state = {k: v.detach().clone() for k, v in state.items()}
        else:
            if self.state.keys() != state.keys():
                raise ValueError('snapshot parameter names changed')
            for k, v in state.items():
                if v.shape != self.state[k].shape or v.dtype != self.state[k].dtype:
                    raise ValueError(f'snapshot tensor changed: {k}')
                if v.is_floating_point() and self.decay:
                    self.state[k].lerp_(v, 1 - self.decay)
                else:
                    self.state[k].copy_(v)
        self.step = step
        return self.state
