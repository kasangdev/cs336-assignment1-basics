import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_values = x.amax(dim=dim, keepdim=True)
    shifted_x = x - max_values
    exp_values = shifted_x.exp()
    sum_values = exp_values.sum(dim=dim, keepdim=True)
    return exp_values / sum_values
