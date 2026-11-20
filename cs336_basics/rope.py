import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None):
        super().__init__()
        self.d_k = d_k
        positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)
        pair_indices = torch.arange(0, (d_k // 2), device=device, dtype=torch.float32)
        inv_freq = 1 / (theta ** (pair_indices * 2 / d_k))
        angles = torch.outer(positions, inv_freq)
        angles_cos = angles.cos()
        angles_sin = angles.sin()
        self.register_buffer("cos_cached", angles_cos, persistent=False)
        self.register_buffer("sin_cached", angles_sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        cos_values = self.cos_cached[token_positions]
        sin_values = self.sin_cached[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos_values - x_odd * sin_values
        rotated_odd = x_even * sin_values + x_odd * cos_values
        result = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(start_dim=-2)
        return result.to(in_dtype)
