from einops import einsum
import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None, ):
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
