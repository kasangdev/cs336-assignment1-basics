import numpy as np
import torch


def get_batch(dataset: np.ndarray,
              batch_size: int,
              context_length: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = len(dataset)
    start_indices = np.random.randint(low=0, high=num_tokens - context_length, size=batch_size)
    input_sequences = []
    target_sequences = []
    for start in start_indices:
        input_sequences.append(dataset[start: start + context_length])
        target_sequences.append(dataset[start + 1: start + context_length + 1])
    inputs_array = np.stack(input_sequences, axis=0)
    targets_array = np.stack(target_sequences, axis=0)
    inputs = torch.tensor(inputs_array, dtype=torch.long, device=device)
    targets = torch.tensor(targets_array, dtype=torch.long, device=device)
    return inputs, targets
