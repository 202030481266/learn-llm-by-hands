import torch
from torch import nn


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) # [dim]
        self.eps = eps
    
    def _norm(self, x: torch.Tensor):
        # [B, S, H, D] / [B, S, H, 1] => [B, S, H, D]， 广播
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * (self._norm(x.float()).type_as(x))
