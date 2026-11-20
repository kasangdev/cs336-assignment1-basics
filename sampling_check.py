import torch
from cs336_basics.generation import sample_next_token

toy_probs = torch.tensor([0.05, 0.15, 0.50, 0.30])
toy_logits = torch.log(toy_probs)
for p in [0.4, 0.7, 1.0]:
    torch.manual_seed(42)
    samples = []
    for _ in range(1000):
        token_tensor = sample_next_token(toy_logits, temperature=1.0, top_p=p)
        samples.append(token_tensor.item())
    print(f"top_p={p}, 集合: {set(samples)}")

for temp in [0.5, 2.0]:
    torch.manual_seed(42)
    samples = []
    for _ in range(1000):
        token_tensor = sample_next_token(toy_logits, temperature=temp, top_p=1.0)
        samples.append(token_tensor.item())
    freq = samples.count(2) / 1000
    print(f"temp={temp}, token 2 的频率: {freq}")
