"""LEACE: closed-form least-squares-minimal linear concept erasure.

Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman (NeurIPS 2023).

Given representations X (N, D) and categorical labels Z (N,), build an affine map
    r ↦ r - B Bp (r - μ)
such that cov(eraser(X), one_hot(Z)) = 0, with minimum mean squared change to X
in the Mahalanobis (cov-X) metric.

Construction:
    Σ_XX = cov(X), Σ_XZ = cov(X, one_hot(Z))
    U    = Σ_XX^{-1/2} Σ_XZ                # whitened cross-cov
    A    = col-span basis of U  (via SVD, with numerical rank threshold)
    B    = Σ_XX^{1/2} A,    Bp = A.T Σ_XX^{-1/2}
The eraser is (I - B Bp) shifted around the mean; the numerical rank of U is
the number of directions actually scrubbed (≤ min(D, K-1)).
"""
import torch


class LeaceEraser:
    def __init__(self, mu: torch.Tensor, B: torch.Tensor, Bp: torch.Tensor):
        self.mu = mu   # (D,)
        self.B = B     # (D, r)
        self.Bp = Bp   # (r, D)

    @classmethod
    def fit(cls, X: torch.Tensor, Z: torch.Tensor, eps: float = 1e-6,
            sigma_reg: float = 0.0) -> "LeaceEraser":
        """Fit on (N, D) representations and (N,) integer labels. Returns an
        eraser usable at any later call site. eps thresholds singular values
        relative to the largest; sigma_reg adds a ridge to Σ_XX for stability.
        """
        orig_dtype = X.dtype
        X = X.to(torch.float64)
        N, D = X.shape
        K = int(Z.max().item()) + 1
        Z_oh = torch.zeros(N, K, dtype=torch.float64, device=X.device)
        Z_oh.scatter_(1, Z.long().unsqueeze(1), 1.0)

        mu_X = X.mean(0)
        Xc = X - mu_X
        Zc = Z_oh - Z_oh.mean(0)

        Sxx = (Xc.T @ Xc) / N
        Sxz = (Xc.T @ Zc) / N
        if sigma_reg > 0:
            Sxx = Sxx + sigma_reg * torch.eye(D, dtype=Sxx.dtype, device=Sxx.device)

        evals, evecs = torch.linalg.eigh(Sxx)
        evals = evals.clamp(min=eps * evals.max())
        sqrt_S = (evecs * evals.sqrt()) @ evecs.T
        invsqrt_S = (evecs * evals.rsqrt()) @ evecs.T

        U = invsqrt_S @ Sxz                          # (D, K)
        U_, S, _ = torch.linalg.svd(U, full_matrices=False)
        s_max = S.max() if S.numel() else torch.tensor(0.0, device=X.device)
        rank = int((S > eps * s_max).sum().item()) if s_max > 0 else 0
        A = U_[:, :rank]                             # (D, rank)

        B = sqrt_S @ A                               # (D, rank)
        Bp = A.T @ invsqrt_S                         # (rank, D)

        return cls(mu_X.to(orig_dtype), B.to(orig_dtype), Bp.to(orig_dtype))

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Apply to last-dim D. X can be any shape (..., D)."""
        mu = self.mu.to(X.dtype)
        B = self.B.to(X.dtype)
        Bp = self.Bp.to(X.dtype)
        return X - (X - mu) @ Bp.T @ B.T

    @property
    def rank(self) -> int:
        return self.B.shape[1]

    def to(self, device) -> "LeaceEraser":
        return LeaceEraser(self.mu.to(device), self.B.to(device), self.Bp.to(device))
