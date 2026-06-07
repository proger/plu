from __future__ import annotations

import torch
from torch import Tensor


def _batch_bool_column(mask: Tensor | None, batch: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.zeros((batch, 1), device=device, dtype=torch.bool)
    return mask.to(device=device, dtype=torch.bool).view(batch, 1)


def apply_sampling_constraints_(
    logits: Tensor,
    *,
    timestamp_begin: int,
    suppress_tokens: Tensor | None = None,
    timestamps_after_sentence_end: bool = False,
    timestamp_token_ids: Tensor | None = None,
    min_timestamp_tokens: Tensor | None = None,
    force_timestamp: Tensor | None = None,
    pair_timestamp: Tensor | None = None,
) -> Tensor:
    """Apply Whisper decode constraints in place and return `logits`.

    Normal transcription suppresses timestamp tokens. Sentence-timestamp mode
    suppresses timestamps except on forced timestamp or paired timestamp steps.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [batch, vocab], got {tuple(logits.shape)}")
    batch, vocab = logits.shape
    if not 0 <= timestamp_begin <= vocab:
        raise ValueError(f"timestamp_begin must be in [0, {vocab}], got {timestamp_begin}")

    if suppress_tokens is not None and suppress_tokens.numel():
        logits.index_fill_(1, suppress_tokens.to(device=logits.device, dtype=torch.long), -float("inf"))

    if not timestamps_after_sentence_end:
        logits[:, timestamp_begin:].fill_(-float("inf"))
        return logits

    timestamp_logits = logits[:, timestamp_begin:]
    if timestamp_token_ids is None:
        timestamp_token_ids = torch.arange(timestamp_begin, vocab, device=logits.device, dtype=torch.long)
    else:
        timestamp_token_ids = timestamp_token_ids.to(device=logits.device, dtype=torch.long)
    if timestamp_token_ids.numel() != vocab - timestamp_begin:
        raise ValueError(
            f"timestamp_token_ids must have {vocab - timestamp_begin} elements, got {timestamp_token_ids.numel()}"
        )

    if min_timestamp_tokens is not None:
        min_timestamp_tokens = min_timestamp_tokens.to(device=logits.device, dtype=torch.long).view(batch)
        valid_timestamps = timestamp_token_ids[None, :] >= min_timestamp_tokens[:, None]
        timestamp_logits = timestamp_logits.masked_fill(~valid_timestamps, -float("inf"))

    force = _batch_bool_column(force_timestamp, batch, logits.device)
    pair = _batch_bool_column(pair_timestamp, batch, logits.device) & ~force
    timestamp_only = force | pair
    logits[:, :timestamp_begin] = torch.where(
        timestamp_only,
        torch.full_like(logits[:, :timestamp_begin], -float("inf")),
        logits[:, :timestamp_begin],
    )
    logits[:, timestamp_begin:] = torch.where(
        timestamp_only,
        timestamp_logits,
        torch.full_like(timestamp_logits, -float("inf")),
    )
    return logits


def sample_next_token(
    logits: Tensor,
    *,
    timestamp_begin: int,
    suppress_tokens: Tensor | None = None,
    timestamps_after_sentence_end: bool = False,
    timestamp_token_ids: Tensor | None = None,
    min_timestamp_tokens: Tensor | None = None,
    force_timestamp: Tensor | None = None,
    pair_timestamp: Tensor | None = None,
    pair_timestamp_tokens: Tensor | None = None,
    greedy_batch_mask: Tensor | None = None,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Sample or greedily select the next token from constrained logits.

    `logits` is modified in place. When `greedy_batch_mask` is provided, rows
    with False entries sample with Gumbel-max at `temperature`; rows with True
    entries use argmax. Forced timestamp rows also use argmax, and paired
    timestamp rows are overwritten with `pair_timestamp_tokens`.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    logits = apply_sampling_constraints_(
        logits,
        timestamp_begin=timestamp_begin,
        suppress_tokens=suppress_tokens,
        timestamps_after_sentence_end=timestamps_after_sentence_end,
        timestamp_token_ids=timestamp_token_ids,
        min_timestamp_tokens=min_timestamp_tokens,
        force_timestamp=force_timestamp,
        pair_timestamp=pair_timestamp,
    )
    logprobs = logits.log_softmax(dim=-1)
    greedy_next_token = logits.argmax(dim=-1)

    if greedy_batch_mask is None:
        next_token = greedy_next_token
    else:
        sample_logits = logits if temperature == 1.0 else logits / temperature
        uniform = torch.rand_like(sample_logits)
        gumbel = -torch.log(-torch.log(uniform.clamp_(min=1e-6, max=1.0 - 1e-6)))
        sampled_next_token = (sample_logits + gumbel).argmax(dim=-1)
        greedy_mask = greedy_batch_mask.to(device=logits.device, dtype=torch.bool)
        if force_timestamp is not None:
            greedy_mask = greedy_mask | force_timestamp.to(device=logits.device, dtype=torch.bool)
        next_token = torch.where(greedy_mask, greedy_next_token, sampled_next_token)

    if timestamps_after_sentence_end and pair_timestamp is not None and pair_timestamp_tokens is not None:
        pair_mask = pair_timestamp.to(device=logits.device, dtype=torch.bool)
        if force_timestamp is not None:
            pair_mask = pair_mask & ~force_timestamp.to(device=logits.device, dtype=torch.bool)
        next_token = torch.where(pair_mask, pair_timestamp_tokens.to(device=logits.device, dtype=torch.long), next_token)

    selected_logprob = logprobs.gather(1, next_token[:, None]).squeeze(1)
    return next_token, selected_logprob


def update_decode_control_(
    next_token: Tensor,
    selected_logprob: Tensor,
    generated_tokens: Tensor,
    generated_logprobs: Tensor,
    generated_lengths: Tensor,
    finished: Tensor,
    current_token: Tensor,
    force_timestamp: Tensor,
    pair_timestamp: Tensor,
    pair_timestamp_tokens: Tensor,
    min_timestamp_tokens: Tensor,
    position: Tensor,
    sampling_seed: Tensor,
    all_finished: Tensor,
    sentence_end_token_mask: Tensor,
    *,
    eos_token_id: int,
    timestamp_begin: int,
    timestamps_after_sentence_end: bool,
    stop_after_first_sentence: bool,
) -> None:
    """Reference in-place update for one decode control-loop step."""
    batch = next_token.numel()
    if selected_logprob.numel() != batch:
        raise ValueError("selected_logprob must have one element per batch row")
    if generated_tokens.ndim != 2 or generated_logprobs.shape != generated_tokens.shape:
        raise ValueError("generated_tokens and generated_logprobs must have matching [batch, max_steps] shape")
    if generated_tokens.shape[0] != batch:
        raise ValueError("generated buffer batch size must match next_token")

    max_steps = generated_tokens.shape[1]
    for index in range(batch):
        next_position = int(position[index].item()) + 1
        was_finished = bool(finished[index].item())
        token = int(next_token[index].item())
        if was_finished:
            current_token[index] = eos_token_id
            force_timestamp[index] = False
            pair_timestamp[index] = False
            position[index] = next_position
            continue

        length = int(generated_lengths[index].item())
        previous_token = int(generated_tokens[index, length - 1].item()) if length > 0 else eos_token_id
        if length < max_steps:
            generated_tokens[index, length] = token
            generated_logprobs[index, length] = selected_logprob[index]
            generated_lengths[index] = length + 1

        row_finished = token == eos_token_id
        if token >= timestamp_begin:
            repeated_sentence_timestamp = (
                stop_after_first_sentence
                and timestamps_after_sentence_end
                and length > 0
                and previous_token == token
            )
            if length > 0 and previous_token < timestamp_begin and previous_token != eos_token_id:
                min_timestamp_tokens[index] = torch.maximum(min_timestamp_tokens[index], next_token[index])
                pair_timestamp[index] = True
                pair_timestamp_tokens[index] = token
            else:
                min_timestamp_tokens[index] = torch.maximum(min_timestamp_tokens[index], next_token[index] + 1)
                pair_timestamp[index] = False
            force_timestamp[index] = False
            if repeated_sentence_timestamp:
                row_finished = True
        elif timestamps_after_sentence_end and bool(sentence_end_token_mask[token].item()):
            force_timestamp[index] = True
            pair_timestamp[index] = False
        else:
            force_timestamp[index] = False
            pair_timestamp[index] = False

        finished[index] = row_finished
        current_token[index] = eos_token_id if row_finished else token
        position[index] = next_position

    sampling_seed.add_(1)
    all_finished.fill_(bool(finished.all().item()))
