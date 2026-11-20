from torch import nn
import torch

from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.feedforward import SiLUFFN, SwiGLU
from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.embedding import Embedding


TRANSFORMER_VARIANTS = ("baseline", "no_rmsnorm", "post_norm", "no_rope", "silu")


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope: RotaryPositionalEmbedding | None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 variant: str = "baseline"):
        super().__init__()
        if variant not in TRANSFORMER_VARIANTS:
            raise ValueError(f"Unknown Transformer variant {variant!r}. Expected one of {TRANSFORMER_VARIANTS}.")
        self.variant = variant
        self.ln1 = (
            nn.Identity() if variant == "no_rmsnorm" else RMSNorm(d_model, device=device, dtype=dtype)
        )
        self.attn = MultiHeadSelfAttention(d_model,
                                           num_heads,
                                           rope=None if variant == "no_rope" else rope,
                                           device=device,
                                           dtype=dtype)
        self.ln2 = (
            nn.Identity() if variant == "no_rmsnorm" else RMSNorm(d_model, device=device, dtype=dtype)
        )

        ffn_class = SiLUFFN if variant == "silu" else SwiGLU
        self.ffn = ffn_class(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        if self.variant == "post_norm":
            after_attention = self.ln1(x + self.attn(x, token_positions))
            output = self.ln2(after_attention + self.ffn(after_attention))
            return output
        after_attention = x + self.attn(self.ln1(x), token_positions)
        output = after_attention + self.ffn(self.ln2(after_attention))
        return output


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 variant: str = "baseline"):
        super().__init__()
        if variant not in TRANSFORMER_VARIANTS:
            raise ValueError(f"Unknown Transformer variant {variant!r}. Expected one of {TRANSFORMER_VARIANTS}.")
        d_k = d_model // num_heads
        rope = None
        if variant != "no_rope":
            rope = RotaryPositionalEmbedding(rope_theta,
                                             d_k,
                                             context_length,
                                             device=device)
        self.token_embeddings = Embedding(vocab_size,
                                          d_model,
                                          device=device,
                                          dtype=dtype)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            transformer_block = TransformerBlock(d_model,
                                                 num_heads,
                                                 d_ff,
                                                 rope=rope,
                                                 device=device,
                                                 dtype=dtype,
                                                 variant=variant)
            self.layers.append(transformer_block)
        self.ln_final = (
            nn.Identity() if variant == "no_rmsnorm" else RMSNorm(d_model, device=device, dtype=dtype)
        )
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor):
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
