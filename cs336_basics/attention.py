from torch import nn
import torch
from einops import einsum, rearrange
import math

from cs336_basics.linear import Linear
from cs336_basics.activations import softmax

from cs336_basics.rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(Q: torch.Tensor,
                                 K: torch.Tensor,
                                 V: torch.Tensor,
                                 mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -math.inf)
    attention_weights = softmax(scores, dim=-1)
    return einsum(attention_weights, V, "... q k, ... k d_v -> ... q d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 rope: RotaryPositionalEmbedding | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self,
                x: torch.Tensor,
                token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        K = rearrange(K, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        V = rearrange(V, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        seq_len = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.unsqueeze(-2)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        seq_positions = torch.arange(seq_len, device=x.device)
        row_indices = seq_positions.unsqueeze(1)
        col_indices = seq_positions.unsqueeze(0)
        causal_mask = (row_indices >= col_indices)
        attention_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        merged = rearrange(attention_output, "... h seq d_k -> ... seq (h d_k)")
        return self.output_proj(merged)
