import torch
from collections.abc import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = []
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad)
    if len(grads) == 0:
        return None
    with torch.no_grad():
        total_square_norm = 0
        for grad in grads:
            total_square_norm += grad.square().sum()
        total_norm = total_square_norm.sqrt()
        eps = 1e-6
        if total_norm < max_l2_norm:
            clip_coef = 1.0
        else:
            clip_coef = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(clip_coef)
