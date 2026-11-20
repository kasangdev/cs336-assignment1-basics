from cs336_basics.optimizers import SGD
import torch

learning_rates = [10.0, 100.0, 1000.0]

for lr in learning_rates:
    torch.manual_seed(42)
    weights = torch.nn.Parameter(torch.randn(10, 10) * 5)
    opt = SGD([weights], lr=lr)
    for step_idx in range(10):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(f"lr={lr}, step={step_idx}, loss={loss.item():.4f}")
        loss.backward()
        opt.step()
