from __future__ import annotations

import pytest
import torch

from plu.ref.sampling import apply_sampling_constraints_
from plu.ref.sampling import sample_next_token as ref_sample_next_token
from plu.ref.sampling import update_decode_control_ as ref_update_decode_control_
from plu.triton.sampling import make_suppress_mask, sample_next_token
from plu.triton.sampling import update_decode_control_ as triton_update_decode_control_


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference_logprob_for_tokens(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    timestamp_begin: int,
    suppress_tokens: torch.Tensor | None = None,
    timestamps_after_sentence_end: bool = False,
    min_timestamp_tokens: torch.Tensor | None = None,
    force_timestamp: torch.Tensor | None = None,
    pair_timestamp: torch.Tensor | None = None,
) -> torch.Tensor:
    constrained = apply_sampling_constraints_(
        logits.clone(),
        timestamp_begin=timestamp_begin,
        suppress_tokens=suppress_tokens,
        timestamps_after_sentence_end=timestamps_after_sentence_end,
        min_timestamp_tokens=min_timestamp_tokens,
        force_timestamp=force_timestamp,
        pair_timestamp=pair_timestamp,
    )
    return constrained.log_softmax(dim=-1).gather(1, token_ids[:, None]).squeeze(1)


def test_greedy_sampling_matches_reference_with_constraints():
    logits = torch.randn(3, 37, device="cuda", dtype=torch.float32)
    timestamp_begin = 24
    suppress_tokens = torch.tensor([2, 5, 30], device="cuda")
    suppress_mask = make_suppress_mask(logits.shape[1], suppress_tokens, device=logits.device)
    force_timestamp = torch.tensor([False, True, False], device="cuda")
    pair_timestamp = torch.tensor([False, False, True], device="cuda")
    pair_timestamp_tokens = torch.tensor([timestamp_begin, timestamp_begin + 3, timestamp_begin + 2], device="cuda")
    min_timestamp_tokens = torch.tensor([timestamp_begin, timestamp_begin + 1, timestamp_begin + 2], device="cuda")
    greedy_batch_mask = torch.tensor([True, True, True], device="cuda")
    seed = torch.tensor(123, device="cuda", dtype=torch.int64)

    ref_next, ref_logprob = ref_sample_next_token(
        logits.clone(),
        timestamp_begin=timestamp_begin,
        suppress_tokens=suppress_tokens,
        timestamps_after_sentence_end=True,
        min_timestamp_tokens=min_timestamp_tokens,
        force_timestamp=force_timestamp,
        pair_timestamp=pair_timestamp,
        pair_timestamp_tokens=pair_timestamp_tokens,
        greedy_batch_mask=greedy_batch_mask,
    )
    tri_next, tri_logprob = sample_next_token(
        logits.clone(),
        timestamp_begin=timestamp_begin,
        suppress_mask=suppress_mask,
        timestamps_after_sentence_end=True,
        min_timestamp_tokens=min_timestamp_tokens,
        force_timestamp=force_timestamp,
        pair_timestamp=pair_timestamp,
        pair_timestamp_tokens=pair_timestamp_tokens,
        greedy_batch_mask=greedy_batch_mask,
        seed=seed,
        block_size=16,
    )

    assert tri_next.tolist() == ref_next.tolist()
    assert torch.allclose(tri_logprob, ref_logprob, atol=1e-5, rtol=1e-5)


def test_greedy_sampling_matches_reference_without_timestamps():
    logits = torch.randn(4, 53, device="cuda", dtype=torch.float32)
    timestamp_begin = 41
    suppress_tokens = torch.tensor([0, 3, 17], device="cuda")
    suppress_mask = make_suppress_mask(logits.shape[1], suppress_tokens, device=logits.device)
    greedy_batch_mask = torch.ones(logits.shape[0], device="cuda", dtype=torch.bool)

    ref_next, ref_logprob = ref_sample_next_token(
        logits.clone(),
        timestamp_begin=timestamp_begin,
        suppress_tokens=suppress_tokens,
        greedy_batch_mask=greedy_batch_mask,
    )
    tri_next, tri_logprob = sample_next_token(
        logits.clone(),
        timestamp_begin=timestamp_begin,
        suppress_mask=suppress_mask,
        greedy_batch_mask=greedy_batch_mask,
        seed=torch.tensor(11, device="cuda", dtype=torch.int64),
        block_size=32,
    )

    assert tri_next.tolist() == ref_next.tolist()
    assert torch.allclose(tri_logprob, ref_logprob, atol=1e-5, rtol=1e-5)


def test_sampled_rows_obey_masks_and_forced_timestamps_are_greedy():
    logits = torch.full((2, 64), -8.0, device="cuda", dtype=torch.float32)
    timestamp_begin = 48
    logits[:, 3] = 12.0
    logits[0, 10] = 9.0
    logits[1, timestamp_begin + 4] = 5.0
    logits[1, timestamp_begin + 5] = 3.0
    suppress_tokens = torch.tensor([3], device="cuda")
    suppress_mask = make_suppress_mask(logits.shape[1], suppress_tokens, device=logits.device)
    force_timestamp = torch.tensor([False, True], device="cuda")
    pair_timestamp = torch.tensor([False, False], device="cuda")
    min_timestamp_tokens = torch.tensor([timestamp_begin, timestamp_begin], device="cuda")
    greedy_batch_mask = torch.tensor([False, False], device="cuda")
    seed = torch.tensor(7, device="cuda", dtype=torch.int64)

    next_token, logprob = sample_next_token(
        logits,
        timestamp_begin=timestamp_begin,
        suppress_mask=suppress_mask,
        timestamps_after_sentence_end=True,
        min_timestamp_tokens=min_timestamp_tokens,
        force_timestamp=force_timestamp,
        pair_timestamp=pair_timestamp,
        greedy_batch_mask=greedy_batch_mask,
        seed=seed,
        block_size=32,
    )

    assert next_token[0].item() < timestamp_begin
    assert next_token[0].item() != 3
    assert next_token[1].item() == timestamp_begin + 4
    ref_logprob = _reference_logprob_for_tokens(
        logits,
        next_token,
        timestamp_begin=timestamp_begin,
        suppress_tokens=suppress_tokens,
        timestamps_after_sentence_end=True,
        min_timestamp_tokens=min_timestamp_tokens,
        force_timestamp=force_timestamp,
        pair_timestamp=pair_timestamp,
    )
    assert torch.allclose(logprob, ref_logprob, atol=1e-5, rtol=1e-5)


def _clone_control_state(state: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {name: tensor.clone().to(device=device) for name, tensor in state.items()}


def _assert_control_states_equal(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name].detach().cpu()
        expected_tensor = expected_tensor.detach().cpu()
        if expected_tensor.dtype.is_floating_point:
            assert torch.allclose(actual_tensor, expected_tensor, atol=1e-6, rtol=1e-6), name
        else:
            assert actual_tensor.tolist() == expected_tensor.tolist(), name


def test_control_update_matches_reference_for_sentence_timestamp_state():
    base_state = {
        "generated_tokens": torch.tensor([[3, -1, -1, -1], [4, 12, -1, -1], [-1, -1, -1, -1]], dtype=torch.long),
        "generated_logprobs": torch.zeros((3, 4), dtype=torch.float32),
        "generated_lengths": torch.tensor([1, 2, 0], dtype=torch.long),
        "finished": torch.tensor([False, False, True], dtype=torch.bool),
        "current_token": torch.zeros(3, dtype=torch.long),
        "force_timestamp": torch.tensor([False, True, False], dtype=torch.bool),
        "pair_timestamp": torch.zeros(3, dtype=torch.bool),
        "pair_timestamp_tokens": torch.full((3,), 10, dtype=torch.long),
        "min_timestamp_tokens": torch.tensor([10, 12, 10], dtype=torch.long),
        "position": torch.full((3,), 6, dtype=torch.long),
        "sampling_seed": torch.full((), 122, dtype=torch.long),
        "all_finished": torch.zeros((), dtype=torch.bool),
        "sentence_end_token_mask": torch.zeros(20, dtype=torch.bool),
    }
    base_state["sentence_end_token_mask"][3] = True

    next_token = torch.tensor([12, 12, 5], dtype=torch.long)
    selected_logprob = torch.tensor([-0.25, -0.5, -2.0], dtype=torch.float32)
    expected = _clone_control_state(base_state, "cpu")
    actual = _clone_control_state(base_state, "cuda")

    ref_update_decode_control_(
        next_token,
        selected_logprob,
        expected["generated_tokens"],
        expected["generated_logprobs"],
        expected["generated_lengths"],
        expected["finished"],
        expected["current_token"],
        expected["force_timestamp"],
        expected["pair_timestamp"],
        expected["pair_timestamp_tokens"],
        expected["min_timestamp_tokens"],
        expected["position"],
        expected["sampling_seed"],
        expected["all_finished"],
        expected["sentence_end_token_mask"],
        eos_token_id=9,
        timestamp_begin=10,
        timestamps_after_sentence_end=True,
        stop_after_first_sentence=True,
    )
    triton_update_decode_control_(
        next_token.to(device="cuda"),
        selected_logprob.to(device="cuda"),
        actual["generated_tokens"],
        actual["generated_logprobs"],
        actual["generated_lengths"],
        actual["finished"],
        actual["current_token"],
        actual["force_timestamp"],
        actual["pair_timestamp"],
        actual["pair_timestamp_tokens"],
        actual["min_timestamp_tokens"],
        actual["position"],
        actual["sampling_seed"],
        actual["all_finished"],
        actual["sentence_end_token_mask"],
        eos_token_id=9,
        timestamp_begin=10,
        timestamps_after_sentence_end=True,
        stop_after_first_sentence=True,
    )
    torch.cuda.synchronize()

    _assert_control_states_equal(actual, expected)


def test_control_update_matches_reference_without_timestamp_mode():
    base_state = {
        "generated_tokens": torch.full((2, 3), -1, dtype=torch.long),
        "generated_logprobs": torch.zeros((2, 3), dtype=torch.float32),
        "generated_lengths": torch.tensor([0, 0], dtype=torch.long),
        "finished": torch.tensor([False, False], dtype=torch.bool),
        "current_token": torch.zeros(2, dtype=torch.long),
        "force_timestamp": torch.ones(2, dtype=torch.bool),
        "pair_timestamp": torch.ones(2, dtype=torch.bool),
        "pair_timestamp_tokens": torch.full((2,), 10, dtype=torch.long),
        "min_timestamp_tokens": torch.full((2,), 10, dtype=torch.long),
        "position": torch.full((2,), 2, dtype=torch.long),
        "sampling_seed": torch.full((), 54, dtype=torch.long),
        "all_finished": torch.zeros((), dtype=torch.bool),
        "sentence_end_token_mask": torch.zeros(20, dtype=torch.bool),
    }
    next_token = torch.tensor([4, 9], dtype=torch.long)
    selected_logprob = torch.tensor([-0.1, -0.2], dtype=torch.float32)
    expected = _clone_control_state(base_state, "cpu")
    actual = _clone_control_state(base_state, "cuda")

    ref_update_decode_control_(
        next_token,
        selected_logprob,
        expected["generated_tokens"],
        expected["generated_logprobs"],
        expected["generated_lengths"],
        expected["finished"],
        expected["current_token"],
        expected["force_timestamp"],
        expected["pair_timestamp"],
        expected["pair_timestamp_tokens"],
        expected["min_timestamp_tokens"],
        expected["position"],
        expected["sampling_seed"],
        expected["all_finished"],
        expected["sentence_end_token_mask"],
        eos_token_id=9,
        timestamp_begin=10,
        timestamps_after_sentence_end=False,
        stop_after_first_sentence=False,
    )
    triton_update_decode_control_(
        next_token.to(device="cuda"),
        selected_logprob.to(device="cuda"),
        actual["generated_tokens"],
        actual["generated_logprobs"],
        actual["generated_lengths"],
        actual["finished"],
        actual["current_token"],
        actual["force_timestamp"],
        actual["pair_timestamp"],
        actual["pair_timestamp_tokens"],
        actual["min_timestamp_tokens"],
        actual["position"],
        actual["sampling_seed"],
        actual["all_finished"],
        actual["sentence_end_token_mask"],
        eos_token_id=9,
        timestamp_begin=10,
        timestamps_after_sentence_end=False,
        stop_after_first_sentence=False,
    )
    torch.cuda.synchronize()

    _assert_control_states_equal(actual, expected)
