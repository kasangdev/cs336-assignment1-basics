import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        if closure is not None:
            loss = closure()
        else:
            loss = None
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if betas[0] < 0.0 or betas[0] >= 1:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if betas[1] < 0.0 or betas[1] >= 1:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        if closure is not None:
            loss = closure()
        else:
            loss = None
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 0
                m = state["m"]
                v = state["v"]
                t = state["t"]
                grad = p.grad.data
                p.data = p.data - lr * weight_decay * p.data
                t += 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                adjusted_lr = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p.data = p.data - adjusted_lr * (m / (v.sqrt() + eps))
                state["m"] = m
                state["v"] = v
                state["t"] = t
        return loss
