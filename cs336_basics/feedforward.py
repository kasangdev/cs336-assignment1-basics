from torch import nn
import torch
from cs336_basics.linear import Linear
from cs336_basics.activations import silu


class SwiGLU(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        value = self.w3(x)
        hidden = gate * value
        return self.w2(hidden)


class SiLUFFN(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = silu(self.w1(x))
        return self.w2(hidden)
