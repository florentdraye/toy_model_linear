"""Label-balanced difference-of-means steering, using the full training support."""
import torch


@torch.no_grad()
def hidden_at(model, edges, depth):
    """Stop at the intervention block; no downstream computation is needed."""
    if not 1 <= depth <= len(model.blocks):
        raise ValueError('depth must be a 1-based transformer block')
    h = model.drop(model.tok_emb(edges) + model.pos_emb)
    for block in model.blocks[:depth]:
        h = block(h)
    return h


class UniformMeanBank:
    """One copy of every allowed trajectory, grouped by its latent label.

    The caller supplies TRAINING support only (not frequency-weighted draws).
    Averaging all paths within each class, then averaging class means, exactly
    integrates the uniform-latent / uniform-conditional-path distribution.
    No random sampling, endpoint labels, optimizer, or amplitude fitting.
    """

    def __init__(self, edges, labels):
        if edges.ndim != 2 or labels.ndim != 1 or len(edges) != len(labels):
            raise ValueError('expected matching [paths, tokens] and [paths] tensors')
        classes, inverse, counts = labels.unique(sorted=True, return_inverse=True,
                                                  return_counts=True)
        if len(classes) < 2:
            raise ValueError('need at least two nonempty latent classes')
        self.classes = classes.tolist()
        self.counts = counts.tolist()
        self.edges = edges[inverse.argsort(stable=True)]

    @torch.no_grad()
    def fit(self, model, targets, depth, positions, batch_size=8192):
        """Return v[k,p] = mu[k,p] - mean_{j != k} mu[j,p], plus all mu.

        Each position has its own vector. Class sizes need not be equal: every
        alternative CLASS receives equal weight, not every negative trajectory.
        FP32 forward passes and FP64 sums avoid low-precision accumulation.
        Model must be in eval mode; its weights and training flag are unchanged.
        """
        if model.training:
            raise ValueError('put the frozen model in eval mode before taking means')
        if batch_size < 1 or not positions or len(set(positions)) != len(positions):
            raise ValueError('need a positive batch size and distinct positions')
        if min(positions) < 0 or max(positions) >= self.edges.shape[1]:
            raise ValueError('position outside the input sequence')
        if any(k not in self.classes for k in targets):
            raise ValueError('target has no paths in the mean bank')
        device = next(model.parameters()).device
        means, start = [], 0
        with torch.autocast(device_type=device.type, enabled=False):
            for count in self.counts:
                total = torch.zeros(len(positions), model.cfg.d_model,
                                    device=device, dtype=torch.float64)
                for edges in self.edges[start:start + count].split(batch_size):
                    h = hidden_at(model, edges.to(device), depth)
                    total += h[:, positions].double().sum(0)
                means.append(total / count)
                start += count
        means = torch.stack(means)
        ix = [self.classes.index(k) for k in targets]
        positive = means[ix]
        negative = (means.sum(0) - positive) / (len(self.classes) - 1)
        return (positive - negative).float(), means

    def description(self, targets):
        counts = dict(zip(self.classes, self.counts))
        return {'method': 'uniform_latent_difference_of_means_v1',
                'positive': 'uniform over all training paths through target',
                'negative': 'equal weight per other latent, uniform paths within latent',
                'support': 'entire training split, no replacement or test paths',
                'alpha': 1., 'forward_dtype': 'float32', 'sum_dtype': 'float64',
                'classes': self.classes, 'class_path_counts': self.counts,
                'total_paths': len(self.edges),
                'positive_paths': [counts[k] for k in targets],
                'negative_paths': [len(self.edges) - counts[k] for k in targets]}
