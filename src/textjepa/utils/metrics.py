"""Collapse diagnostics for latent states."""

import torch


@torch.no_grad()
def feature_std(x: torch.Tensor) -> float:
    """Mean per-dimension std of a [N, D] feature matrix."""
    return x.float().std(dim=0).mean().item()


@torch.no_grad()
def effective_rank(x: torch.Tensor, eps: float = 1e-7) -> float:
    """Entropy-based effective rank (RankMe) of a [N, D] feature matrix."""
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / (s.sum() + eps) + eps
    return torch.exp(-(p * p.log()).sum()).item()
