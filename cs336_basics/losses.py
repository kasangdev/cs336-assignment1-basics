import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits = logits.amax(dim=-1, keepdim=True)
    shifted_logits = logits - max_logits
    log_normalizers = shifted_logits.exp().sum(dim=-1).log()
    targets_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(dim=-1))
    losses = log_normalizers - targets_logits.squeeze(dim=-1)
    loss = losses.mean()
    return loss
