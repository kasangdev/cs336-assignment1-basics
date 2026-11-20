import math

import torch
from cs336_basics.activations import softmax


def sample_next_token(logits, temperature=1.0, top_p=1.0) -> torch.Tensor:
    max_val = logits.amax()
    shifted_logits = logits - max_val
    scaled_logits = shifted_logits / temperature
    probs = softmax(scaled_logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    keep_mask = (cumulative_probs - sorted_probs) < top_p
    filtered_probs = sorted_probs.masked_fill(~keep_mask, 0.0)
    filtered_probs = filtered_probs / filtered_probs.sum()
    sampled_index = torch.multinomial(filtered_probs, num_samples=1)
    next_token = sorted_indices[sampled_index]
    return next_token.to(torch.long)


def _apply_repetition_controls(logits, token_ids, repetition_penalty, no_repeat_ngram_size):
    if repetition_penalty == 1.0 and no_repeat_ngram_size == 0:
        return logits
    adjusted = logits.clone()
    if repetition_penalty != 1.0 and token_ids:
        seen_ids = torch.tensor(sorted(set(token_ids)), dtype=torch.long, device=logits.device)
        seen_scores = adjusted[seen_ids]
        adjusted[seen_ids] = torch.where(
            seen_scores < 0,
            seen_scores * repetition_penalty,
            seen_scores / repetition_penalty,
        )
    if no_repeat_ngram_size > 0:
        prefix_size = no_repeat_ngram_size - 1
        prefix = tuple(token_ids[-prefix_size:]) if prefix_size else ()
        banned = {
            token_ids[start + prefix_size]
            for start in range(len(token_ids) - no_repeat_ngram_size + 1)
            if tuple(token_ids[start:start + prefix_size]) == prefix
        }
        if banned:
            banned_ids = torch.tensor(sorted(banned), dtype=torch.long, device=logits.device)
            previous_scores = adjusted[banned_ids].clone()
            adjusted[banned_ids] = -torch.inf
            if not torch.isfinite(adjusted).any():
                adjusted[banned_ids] = previous_scores
    return adjusted


def generate_tokens(model,
                    prompt_ids,
                    max_new_tokens,
                    context_length,
                    eos_token_id,
                    temperature=1.0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=0) -> list[int]:
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be a positive finite number.")
    if not isinstance(no_repeat_ngram_size, int) or no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be a nonnegative integer.")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generated_ids = prompt_ids.copy()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context_ids = generated_ids[-context_length:]
                input_ids = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
                logits = model(input_ids)
                next_logits = logits[0, -1, :]
                next_logits = _apply_repetition_controls(
                    next_logits, generated_ids, repetition_penalty, no_repeat_ngram_size
                )
                next_token = sample_next_token(next_logits, temperature, top_p)
                next_token_id = next_token.item()
                generated_ids.append(next_token_id)
                if next_token_id == eos_token_id: break
    finally:
        model.train(was_training)
    return generated_ids
