from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


def make_suppress_mask(vocab_size: int, suppress_tokens: Tensor | None, *, device: torch.device | str) -> Tensor:
    mask = torch.zeros(vocab_size, device=device, dtype=torch.bool)
    if suppress_tokens is not None and suppress_tokens.numel():
        mask.index_fill_(0, suppress_tokens.to(device=device, dtype=torch.long), True)
    return mask


@triton.jit
def _constrained_logits(
    logits,
    offsets,
    row: tl.tensor,
    vocab_size: tl.constexpr,
    timestamp_begin: tl.constexpr,
    suppress_mask_ptr,
    min_timestamp_tokens_ptr,
    force_timestamp_ptr,
    pair_timestamp_ptr,
    has_suppress_mask: tl.constexpr,
    has_min_timestamp: tl.constexpr,
    has_force_timestamp: tl.constexpr,
    has_pair_timestamp: tl.constexpr,
    timestamps_after_sentence_end: tl.constexpr,
):
    valid = offsets < vocab_size
    if has_suppress_mask:
        suppressed = tl.load(suppress_mask_ptr + offsets, mask=valid, other=1).to(tl.int1)
        valid = valid & ~suppressed

    is_timestamp = offsets >= timestamp_begin
    if timestamps_after_sentence_end:
        min_timestamp = timestamp_begin
        if has_min_timestamp:
            min_timestamp = tl.load(min_timestamp_tokens_ptr + row)
        force_timestamp = False
        if has_force_timestamp:
            force_timestamp = tl.load(force_timestamp_ptr + row).to(tl.int1)
        pair_timestamp = False
        if has_pair_timestamp:
            pair_timestamp = tl.load(pair_timestamp_ptr + row).to(tl.int1) & ~force_timestamp
        timestamp_only = force_timestamp | pair_timestamp
        valid_timestamp = is_timestamp & (offsets >= min_timestamp)
        valid_text = ~is_timestamp
        valid = valid & tl.where(timestamp_only, valid_timestamp, valid_text)
    else:
        valid = valid & ~is_timestamp

    return tl.where(valid, logits, -float("inf"))


@triton.jit
def _sampling_stage1_kernel(
    logits_ptr,
    suppress_mask_ptr,
    min_timestamp_tokens_ptr,
    force_timestamp_ptr,
    pair_timestamp_ptr,
    seed_ptr,
    partial_logit_max_ptr,
    partial_greedy_value_ptr,
    partial_greedy_index_ptr,
    partial_sample_value_ptr,
    partial_sample_index_ptr,
    vocab_size: tl.constexpr,
    timestamp_begin: tl.constexpr,
    num_blocks: tl.constexpr,
    has_suppress_mask: tl.constexpr,
    has_min_timestamp: tl.constexpr,
    has_force_timestamp: tl.constexpr,
    has_pair_timestamp: tl.constexpr,
    timestamps_after_sentence_end: tl.constexpr,
    temperature: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < vocab_size
    logits = tl.load(logits_ptr + row * vocab_size + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    constrained = _constrained_logits(
        logits,
        offsets,
        row,
        vocab_size,
        timestamp_begin,
        suppress_mask_ptr,
        min_timestamp_tokens_ptr,
        force_timestamp_ptr,
        pair_timestamp_ptr,
        has_suppress_mask,
        has_min_timestamp,
        has_force_timestamp,
        has_pair_timestamp,
        timestamps_after_sentence_end,
    )

    logit_max = tl.max(constrained, axis=0)
    greedy_indices = tl.where(constrained == logit_max, offsets, vocab_size)
    greedy_index = tl.min(greedy_indices, axis=0)

    seed = tl.load(seed_ptr).to(tl.uint32)
    rng_offsets = row * vocab_size + offsets
    uniform = tl.rand(seed, rng_offsets)
    uniform = tl.minimum(tl.maximum(uniform, 1.0e-6), 1.0 - 1.0e-6)
    gumbel = -tl.log(-tl.log(uniform))
    sample_scores = constrained / temperature + gumbel
    sample_max = tl.max(sample_scores, axis=0)
    sample_indices = tl.where(sample_scores == sample_max, offsets, vocab_size)
    sample_index = tl.min(sample_indices, axis=0)

    partial_offset = row * num_blocks + block
    tl.store(partial_logit_max_ptr + partial_offset, logit_max)
    tl.store(partial_greedy_value_ptr + partial_offset, logit_max)
    tl.store(partial_greedy_index_ptr + partial_offset, greedy_index)
    tl.store(partial_sample_value_ptr + partial_offset, sample_max)
    tl.store(partial_sample_index_ptr + partial_offset, sample_index)


@triton.jit
def _sampling_stage2_kernel(
    logits_ptr,
    suppress_mask_ptr,
    min_timestamp_tokens_ptr,
    force_timestamp_ptr,
    pair_timestamp_ptr,
    pair_timestamp_tokens_ptr,
    greedy_batch_mask_ptr,
    partial_logit_max_ptr,
    partial_greedy_value_ptr,
    partial_greedy_index_ptr,
    partial_sample_value_ptr,
    partial_sample_index_ptr,
    next_token_ptr,
    row_max_ptr,
    selected_logit_ptr,
    vocab_size: tl.constexpr,
    timestamp_begin: tl.constexpr,
    num_blocks: tl.constexpr,
    has_suppress_mask: tl.constexpr,
    has_min_timestamp: tl.constexpr,
    has_force_timestamp: tl.constexpr,
    has_pair_timestamp: tl.constexpr,
    has_pair_timestamp_tokens: tl.constexpr,
    has_greedy_batch_mask: tl.constexpr,
    timestamps_after_sentence_end: tl.constexpr,
    reduce_block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, reduce_block)
    mask = offsets < num_blocks

    logit_max_values = tl.load(partial_logit_max_ptr + row * num_blocks + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(logit_max_values, axis=0)

    greedy_values = tl.load(partial_greedy_value_ptr + row * num_blocks + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    greedy_indices = tl.load(partial_greedy_index_ptr + row * num_blocks + offsets, mask=mask, other=2147483647)
    greedy_value = tl.max(greedy_values, axis=0)
    greedy_index = tl.min(tl.where(greedy_values == greedy_value, greedy_indices, 2147483647), axis=0)

    sample_values = tl.load(partial_sample_value_ptr + row * num_blocks + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    sample_indices = tl.load(partial_sample_index_ptr + row * num_blocks + offsets, mask=mask, other=2147483647)
    sample_value = tl.max(sample_values, axis=0)
    sample_index = tl.min(tl.where(sample_values == sample_value, sample_indices, 2147483647), axis=0)

    force_timestamp = False
    if has_force_timestamp:
        force_timestamp = tl.load(force_timestamp_ptr + row).to(tl.int1)
    pair_timestamp = False
    if has_pair_timestamp:
        pair_timestamp = tl.load(pair_timestamp_ptr + row).to(tl.int1) & ~force_timestamp
    greedy = True
    if has_greedy_batch_mask:
        greedy = tl.load(greedy_batch_mask_ptr + row).to(tl.int1) | force_timestamp

    selected = tl.where(greedy, greedy_index, sample_index)
    if has_pair_timestamp_tokens:
        pair_token = tl.load(pair_timestamp_tokens_ptr + row)
        selected = tl.where(pair_timestamp, pair_token, selected)

    selected_valid = selected < vocab_size
    selected_raw = tl.load(logits_ptr + row * vocab_size + selected, mask=selected_valid, other=-float("inf")).to(tl.float32)
    selected_constrained = _constrained_logits(
        selected_raw,
        selected,
        row,
        vocab_size,
        timestamp_begin,
        suppress_mask_ptr,
        min_timestamp_tokens_ptr,
        force_timestamp_ptr,
        pair_timestamp_ptr,
        has_suppress_mask,
        has_min_timestamp,
        has_force_timestamp,
        has_pair_timestamp,
        timestamps_after_sentence_end,
    )

    tl.store(next_token_ptr + row, selected)
    tl.store(row_max_ptr + row, row_max)
    tl.store(selected_logit_ptr + row, selected_constrained)


@triton.jit
def _sampling_stage3_kernel(
    logits_ptr,
    suppress_mask_ptr,
    min_timestamp_tokens_ptr,
    force_timestamp_ptr,
    pair_timestamp_ptr,
    row_max_ptr,
    partial_sum_ptr,
    vocab_size: tl.constexpr,
    timestamp_begin: tl.constexpr,
    num_blocks: tl.constexpr,
    has_suppress_mask: tl.constexpr,
    has_min_timestamp: tl.constexpr,
    has_force_timestamp: tl.constexpr,
    has_pair_timestamp: tl.constexpr,
    timestamps_after_sentence_end: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < vocab_size
    logits = tl.load(logits_ptr + row * vocab_size + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    constrained = _constrained_logits(
        logits,
        offsets,
        row,
        vocab_size,
        timestamp_begin,
        suppress_mask_ptr,
        min_timestamp_tokens_ptr,
        force_timestamp_ptr,
        pair_timestamp_ptr,
        has_suppress_mask,
        has_min_timestamp,
        has_force_timestamp,
        has_pair_timestamp,
        timestamps_after_sentence_end,
    )
    row_max = tl.load(row_max_ptr + row).to(tl.float32)
    values = tl.exp(constrained - row_max)
    values = tl.where(mask, values, 0.0)
    tl.store(partial_sum_ptr + row * num_blocks + block, tl.sum(values, axis=0))


@triton.jit
def _sampling_stage4_kernel(
    partial_sum_ptr,
    row_max_ptr,
    selected_logit_ptr,
    selected_logprob_ptr,
    num_blocks: tl.constexpr,
    reduce_block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, reduce_block)
    mask = offsets < num_blocks
    partial = tl.load(partial_sum_ptr + row * num_blocks + offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.sum(partial, axis=0)
    row_max = tl.load(row_max_ptr + row).to(tl.float32)
    selected_logit = tl.load(selected_logit_ptr + row).to(tl.float32)
    selected_logprob = selected_logit - (tl.log(row_sum) + row_max)
    tl.store(selected_logprob_ptr + row, selected_logprob)


def _optional_cuda_tensor(name: str, tensor: Tensor | None, *, device: torch.device, shape0: int | None = None) -> Tensor | None:
    if tensor is None:
        return None
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if shape0 is not None and tensor.numel() != shape0:
        raise ValueError(f"{name} must have {shape0} elements, got {tensor.numel()}")
    return tensor.contiguous()


def sample_next_token(
    logits: Tensor,
    *,
    timestamp_begin: int,
    suppress_mask: Tensor | None = None,
    timestamps_after_sentence_end: bool = False,
    min_timestamp_tokens: Tensor | None = None,
    force_timestamp: Tensor | None = None,
    pair_timestamp: Tensor | None = None,
    pair_timestamp_tokens: Tensor | None = None,
    greedy_batch_mask: Tensor | None = None,
    seed: Tensor | None = None,
    temperature: float = 1.0,
    block_size: int = 1024,
) -> tuple[Tensor, Tensor]:
    if not logits.is_cuda:
        raise RuntimeError("plu.triton.sampling requires CUDA logits")
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [batch, vocab], got {tuple(logits.shape)}")
    if not logits.is_contiguous():
        raise ValueError("logits must be contiguous")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    batch, vocab_size = logits.shape
    if not 0 <= timestamp_begin <= vocab_size:
        raise ValueError(f"timestamp_begin must be in [0, {vocab_size}], got {timestamp_begin}")
    if suppress_mask is not None:
        if not suppress_mask.is_cuda or suppress_mask.device != logits.device:
            raise RuntimeError("suppress_mask must be a CUDA tensor on the logits device")
        if suppress_mask.numel() != vocab_size:
            raise ValueError(f"suppress_mask must have {vocab_size} elements, got {suppress_mask.numel()}")
        suppress_mask = suppress_mask.contiguous()
    if seed is None:
        seed = torch.zeros((), device=logits.device, dtype=torch.int64)
    else:
        if not seed.is_cuda or seed.device != logits.device or seed.numel() != 1:
            raise ValueError("seed must be a scalar CUDA tensor on the logits device")
        seed = seed.contiguous()

    min_timestamp_tokens = _optional_cuda_tensor("min_timestamp_tokens", min_timestamp_tokens, device=logits.device, shape0=batch)
    force_timestamp = _optional_cuda_tensor("force_timestamp", force_timestamp, device=logits.device, shape0=batch)
    pair_timestamp = _optional_cuda_tensor("pair_timestamp", pair_timestamp, device=logits.device, shape0=batch)
    pair_timestamp_tokens = _optional_cuda_tensor("pair_timestamp_tokens", pair_timestamp_tokens, device=logits.device, shape0=batch)
    greedy_batch_mask = _optional_cuda_tensor("greedy_batch_mask", greedy_batch_mask, device=logits.device, shape0=batch)

    block_size = triton.next_power_of_2(block_size)
    num_blocks = triton.cdiv(vocab_size, block_size)
    reduce_block = triton.next_power_of_2(num_blocks)
    partial_shape = (batch, num_blocks)
    partial_logit_max = torch.empty(partial_shape, device=logits.device, dtype=torch.float32)
    partial_greedy_value = torch.empty(partial_shape, device=logits.device, dtype=torch.float32)
    partial_greedy_index = torch.empty(partial_shape, device=logits.device, dtype=torch.int64)
    partial_sample_value = torch.empty(partial_shape, device=logits.device, dtype=torch.float32)
    partial_sample_index = torch.empty(partial_shape, device=logits.device, dtype=torch.int64)
    partial_sum = torch.empty(partial_shape, device=logits.device, dtype=torch.float32)
    row_max = torch.empty(batch, device=logits.device, dtype=torch.float32)
    selected_logit = torch.empty(batch, device=logits.device, dtype=torch.float32)
    next_token = torch.empty(batch, device=logits.device, dtype=torch.long)
    selected_logprob = torch.empty(batch, device=logits.device, dtype=torch.float32)

    has_suppress_mask = suppress_mask is not None
    has_min_timestamp = min_timestamp_tokens is not None
    has_force_timestamp = force_timestamp is not None
    has_pair_timestamp = pair_timestamp is not None
    has_pair_timestamp_tokens = pair_timestamp_tokens is not None
    has_greedy_batch_mask = greedy_batch_mask is not None
    dummy = logits

    _sampling_stage1_kernel[(batch, num_blocks)](
        logits,
        suppress_mask if suppress_mask is not None else dummy,
        min_timestamp_tokens if min_timestamp_tokens is not None else dummy,
        force_timestamp if force_timestamp is not None else dummy,
        pair_timestamp if pair_timestamp is not None else dummy,
        seed,
        partial_logit_max,
        partial_greedy_value,
        partial_greedy_index,
        partial_sample_value,
        partial_sample_index,
        vocab_size,
        timestamp_begin,
        num_blocks,
        has_suppress_mask,
        has_min_timestamp,
        has_force_timestamp,
        has_pair_timestamp,
        timestamps_after_sentence_end,
        float(temperature),
        block_size,
        num_warps=4,
    )
    _sampling_stage2_kernel[(batch,)](
        logits,
        suppress_mask if suppress_mask is not None else dummy,
        min_timestamp_tokens if min_timestamp_tokens is not None else dummy,
        force_timestamp if force_timestamp is not None else dummy,
        pair_timestamp if pair_timestamp is not None else dummy,
        pair_timestamp_tokens if pair_timestamp_tokens is not None else dummy,
        greedy_batch_mask if greedy_batch_mask is not None else dummy,
        partial_logit_max,
        partial_greedy_value,
        partial_greedy_index,
        partial_sample_value,
        partial_sample_index,
        next_token,
        row_max,
        selected_logit,
        vocab_size,
        timestamp_begin,
        num_blocks,
        has_suppress_mask,
        has_min_timestamp,
        has_force_timestamp,
        has_pair_timestamp,
        has_pair_timestamp_tokens,
        has_greedy_batch_mask,
        timestamps_after_sentence_end,
        reduce_block,
        num_warps=1,
    )
    _sampling_stage3_kernel[(batch, num_blocks)](
        logits,
        suppress_mask if suppress_mask is not None else dummy,
        min_timestamp_tokens if min_timestamp_tokens is not None else dummy,
        force_timestamp if force_timestamp is not None else dummy,
        pair_timestamp if pair_timestamp is not None else dummy,
        row_max,
        partial_sum,
        vocab_size,
        timestamp_begin,
        num_blocks,
        has_suppress_mask,
        has_min_timestamp,
        has_force_timestamp,
        has_pair_timestamp,
        timestamps_after_sentence_end,
        block_size,
        num_warps=4,
    )
    _sampling_stage4_kernel[(batch,)](
        partial_sum,
        row_max,
        selected_logit,
        selected_logprob,
        num_blocks,
        reduce_block,
        num_warps=1,
    )
    return next_token, selected_logprob


@triton.jit
def _decode_control_update_kernel(
    next_token_ptr,
    selected_logprob_ptr,
    generated_tokens_ptr,
    generated_logprobs_ptr,
    generated_lengths_ptr,
    finished_ptr,
    current_token_ptr,
    force_timestamp_ptr,
    pair_timestamp_ptr,
    pair_timestamp_tokens_ptr,
    min_timestamp_tokens_ptr,
    position_ptr,
    sampling_seed_ptr,
    all_finished_ptr,
    sentence_end_token_mask_ptr,
    eos_token_id: tl.constexpr,
    timestamp_begin: tl.constexpr,
    vocab_size: tl.constexpr,
    max_steps: tl.constexpr,
    timestamps_after_sentence_end: tl.constexpr,
    stop_after_first_sentence: tl.constexpr,
    batch: tl.constexpr,
    block_b: tl.constexpr,
):
    offsets = tl.arange(0, block_b)
    active = offsets < batch
    token = tl.load(next_token_ptr + offsets, mask=active, other=eos_token_id)
    logprob = tl.load(selected_logprob_ptr + offsets, mask=active, other=0.0)
    length = tl.load(generated_lengths_ptr + offsets, mask=active, other=0)
    was_finished = tl.load(finished_ptr + offsets, mask=active, other=1).to(tl.int1)
    current_position = tl.load(position_ptr + offsets, mask=active, other=0)
    next_position = current_position + 1
    previous = tl.load(
        generated_tokens_ptr + offsets * max_steps + length - 1,
        mask=active & (length > 0),
        other=eos_token_id,
    )

    append = active & ~was_finished & (length < max_steps)
    write_index = offsets * max_steps + length
    tl.store(generated_tokens_ptr + write_index, token, mask=append)
    tl.store(generated_logprobs_ptr + write_index, logprob, mask=append)
    tl.store(generated_lengths_ptr + offsets, length + 1, mask=append)

    is_timestamp = token >= timestamp_begin
    previous_is_text = (length > 0) & (previous < timestamp_begin) & (previous != eos_token_id)
    repeated_sentence_timestamp = (
        stop_after_first_sentence
        & timestamps_after_sentence_end
        & (length > 0)
        & (previous == token)
        & is_timestamp
    )
    token_is_sentence_end = False
    if timestamps_after_sentence_end:
        token_is_sentence_end = tl.load(
            sentence_end_token_mask_ptr + token,
            mask=active & (token >= 0) & (token < vocab_size),
            other=0,
        ).to(tl.int1)

    old_min_timestamp = tl.load(min_timestamp_tokens_ptr + offsets, mask=active, other=timestamp_begin)
    timestamp_min_from_text = tl.maximum(old_min_timestamp, token)
    timestamp_min_from_timestamp = tl.maximum(old_min_timestamp, token + 1)
    next_min_timestamp = tl.where(previous_is_text, timestamp_min_from_text, timestamp_min_from_timestamp)
    tl.store(min_timestamp_tokens_ptr + offsets, next_min_timestamp, mask=active & ~was_finished & is_timestamp)

    next_pair_timestamp = active & ~was_finished & is_timestamp & previous_is_text
    next_force_timestamp = active & ~was_finished & ~is_timestamp & timestamps_after_sentence_end & token_is_sentence_end
    row_finished = was_finished | (active & ((token == eos_token_id) | repeated_sentence_timestamp))
    next_current_token = tl.where(row_finished, eos_token_id, token)

    tl.store(pair_timestamp_ptr + offsets, next_pair_timestamp, mask=active)
    tl.store(pair_timestamp_tokens_ptr + offsets, token, mask=next_pair_timestamp)
    tl.store(force_timestamp_ptr + offsets, next_force_timestamp, mask=active)
    tl.store(finished_ptr + offsets, row_finished, mask=active)
    tl.store(current_token_ptr + offsets, next_current_token, mask=active)
    tl.store(position_ptr + offsets, next_position, mask=active)

    done_values = tl.where(active, row_finished.to(tl.int32), 1)
    all_done = tl.min(done_values, axis=0) == 1
    sampling_seed = tl.load(sampling_seed_ptr)
    tl.store(sampling_seed_ptr, sampling_seed + 1)
    tl.store(all_finished_ptr, all_done)


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
    tensors = (
        next_token,
        selected_logprob,
        generated_tokens,
        generated_logprobs,
        generated_lengths,
        finished,
        current_token,
        force_timestamp,
        pair_timestamp,
        pair_timestamp_tokens,
        min_timestamp_tokens,
        position,
        sampling_seed,
        all_finished,
        sentence_end_token_mask,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("plu.triton.sampling.update_decode_control_ requires CUDA tensors")
    batch = next_token.numel()
    if selected_logprob.numel() != batch:
        raise ValueError("selected_logprob must have one element per batch row")
    if generated_tokens.ndim != 2 or generated_tokens.shape != generated_logprobs.shape:
        raise ValueError("generated_tokens and generated_logprobs must have matching [batch, max_steps] shape")
    if generated_tokens.shape[0] != batch:
        raise ValueError("generated buffer batch size must match next_token")
    for name, tensor in (
        ("generated_lengths", generated_lengths),
        ("finished", finished),
        ("current_token", current_token),
        ("force_timestamp", force_timestamp),
        ("pair_timestamp", pair_timestamp),
        ("pair_timestamp_tokens", pair_timestamp_tokens),
        ("min_timestamp_tokens", min_timestamp_tokens),
        ("position", position),
    ):
        if tensor.numel() != batch:
            raise ValueError(f"{name} must have {batch} elements, got {tensor.numel()}")
    if sampling_seed.numel() != 1 or all_finished.numel() != 1:
        raise ValueError("sampling_seed and all_finished must be scalar tensors")
    if sentence_end_token_mask.ndim != 1:
        raise ValueError("sentence_end_token_mask must be a 1D tensor")

    block_b = triton.next_power_of_2(batch)
    _decode_control_update_kernel[(1,)](
        next_token,
        selected_logprob,
        generated_tokens,
        generated_logprobs,
        generated_lengths,
        finished,
        current_token,
        force_timestamp,
        pair_timestamp,
        pair_timestamp_tokens,
        min_timestamp_tokens,
        position,
        sampling_seed,
        all_finished,
        sentence_end_token_mask,
        eos_token_id,
        timestamp_begin,
        sentence_end_token_mask.numel(),
        generated_tokens.shape[1],
        timestamps_after_sentence_end,
        stop_after_first_sentence,
        batch,
        block_b,
        num_warps=1,
    )
