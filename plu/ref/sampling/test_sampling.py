from __future__ import annotations

import torch

from plu.ref.sampling import apply_sampling_constraints_, sample_next_token, update_decode_control_


def test_constraints_suppress_timestamps_without_sentence_timestamp_mode():
    logits = torch.zeros((1, 8))
    logits[0, 2] = 1.0
    logits[0, 5] = 9.0

    constrained = apply_sampling_constraints_(logits, timestamp_begin=4)

    assert torch.isfinite(constrained[0, :4]).all()
    assert torch.isneginf(constrained[0, 4:]).all()
    assert int(constrained.argmax(dim=-1).item()) == 2


def test_forced_timestamp_uses_greedy_timestamp_for_all_rows():
    logits = torch.full((2, 8), -10.0)
    logits[:, 1] = 100.0
    logits[0, 5] = 3.0
    logits[0, 6] = 2.0
    logits[1, 4] = 1.0
    logits[1, 7] = 4.0

    next_token, selected_logprob = sample_next_token(
        logits,
        timestamp_begin=4,
        timestamps_after_sentence_end=True,
        min_timestamp_tokens=torch.tensor([4, 4]),
        force_timestamp=torch.tensor([True, True]),
        pair_timestamp=torch.tensor([False, False]),
        greedy_batch_mask=torch.tensor([True, False]),
    )
    expected = torch.stack(
        [
            torch.log_softmax(torch.tensor([3.0, 2.0, -10.0, -10.0]), dim=-1)[0],
            torch.log_softmax(torch.tensor([1.0, -10.0, -10.0, 4.0]), dim=-1)[3],
        ]
    )

    assert next_token.tolist() == [5, 7]
    assert torch.allclose(selected_logprob, expected)


def test_min_timestamp_masks_older_timestamps():
    logits = torch.full((1, 8), -10.0)
    logits[0, 4] = 9.0
    logits[0, 5] = 2.0
    logits[0, 6] = 1.0

    next_token, _ = sample_next_token(
        logits,
        timestamp_begin=4,
        timestamps_after_sentence_end=True,
        min_timestamp_tokens=torch.tensor([5]),
        force_timestamp=torch.tensor([True]),
    )

    assert next_token.tolist() == [5]


def test_pair_timestamp_overrides_sampled_token_and_reports_pair_logprob():
    logits = torch.full((1, 8), -10.0)
    logits[0, 4] = 4.0
    logits[0, 5] = 1.0
    pair_token = torch.tensor([5])

    next_token, selected_logprob = sample_next_token(
        logits,
        timestamp_begin=4,
        timestamps_after_sentence_end=True,
        min_timestamp_tokens=torch.tensor([4]),
        force_timestamp=torch.tensor([False]),
        pair_timestamp=torch.tensor([True]),
        pair_timestamp_tokens=pair_token,
        greedy_batch_mask=torch.tensor([False]),
    )
    expected_logprob = torch.log_softmax(torch.tensor([[4.0, 1.0, -10.0, -10.0]]), dim=-1)[0, 1]

    assert next_token.tolist() == [5]
    assert torch.allclose(selected_logprob, expected_logprob[None])


def test_suppress_tokens_are_removed_before_sampling():
    logits = torch.zeros((1, 8))
    logits[0, 1] = 5.0
    logits[0, 2] = 2.0

    next_token, _ = sample_next_token(logits, timestamp_begin=4, suppress_tokens=torch.tensor([1]))

    assert next_token.tolist() == [2]


def test_update_decode_control_forces_timestamp_after_sentence_end():
    generated_tokens = torch.full((1, 4), -1, dtype=torch.long)
    generated_logprobs = torch.zeros((1, 4))
    generated_lengths = torch.zeros(1, dtype=torch.long)
    finished = torch.zeros(1, dtype=torch.bool)
    current_token = torch.zeros(1, dtype=torch.long)
    force_timestamp = torch.zeros(1, dtype=torch.bool)
    pair_timestamp = torch.zeros(1, dtype=torch.bool)
    pair_timestamp_tokens = torch.zeros(1, dtype=torch.long)
    min_timestamp_tokens = torch.full((1,), 10, dtype=torch.long)
    position = torch.full((1,), 6, dtype=torch.long)
    sampling_seed = torch.full((), 122, dtype=torch.long)
    all_finished = torch.zeros((), dtype=torch.bool)
    sentence_end = torch.zeros(20, dtype=torch.bool)
    sentence_end[3] = True

    update_decode_control_(
        torch.tensor([3]),
        torch.tensor([-0.5]),
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
        sentence_end,
        eos_token_id=9,
        timestamp_begin=10,
        timestamps_after_sentence_end=True,
        stop_after_first_sentence=False,
    )

    assert generated_tokens.tolist() == [[3, -1, -1, -1]]
    assert generated_lengths.tolist() == [1]
    assert current_token.tolist() == [3]
    assert force_timestamp.tolist() == [True]
    assert pair_timestamp.tolist() == [False]
    assert position.tolist() == [7]
    assert sampling_seed.item() == 123
    assert not all_finished.item()


def test_update_decode_control_pairs_and_stops_after_repeated_timestamp():
    generated_tokens = torch.tensor([[3, 12, -1, -1]])
    generated_logprobs = torch.zeros((1, 4))
    generated_lengths = torch.tensor([2])
    finished = torch.zeros(1, dtype=torch.bool)
    current_token = torch.zeros(1, dtype=torch.long)
    force_timestamp = torch.ones(1, dtype=torch.bool)
    pair_timestamp = torch.zeros(1, dtype=torch.bool)
    pair_timestamp_tokens = torch.zeros(1, dtype=torch.long)
    min_timestamp_tokens = torch.full((1,), 12, dtype=torch.long)
    position = torch.full((1,), 7, dtype=torch.long)
    sampling_seed = torch.full((), 123, dtype=torch.long)
    all_finished = torch.zeros((), dtype=torch.bool)
    sentence_end = torch.zeros(20, dtype=torch.bool)

    update_decode_control_(
        torch.tensor([12]),
        torch.tensor([-0.1]),
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
        sentence_end,
        eos_token_id=9,
        timestamp_begin=10,
        timestamps_after_sentence_end=True,
        stop_after_first_sentence=True,
    )

    assert generated_tokens.tolist() == [[3, 12, 12, -1]]
    assert generated_lengths.tolist() == [3]
    assert finished.tolist() == [True]
    assert current_token.tolist() == [9]
    assert pair_timestamp.tolist() == [False]
    assert min_timestamp_tokens.tolist() == [13]
    assert position.tolist() == [8]
    assert sampling_seed.item() == 124
    assert all_finished.item()


def test_update_decode_control_skips_finished_rows():
    generated_tokens = torch.full((2, 3), -1, dtype=torch.long)
    generated_logprobs = torch.zeros((2, 3))
    generated_lengths = torch.tensor([0, 1])
    finished = torch.tensor([False, True])
    current_token = torch.zeros(2, dtype=torch.long)
    force_timestamp = torch.zeros(2, dtype=torch.bool)
    pair_timestamp = torch.zeros(2, dtype=torch.bool)
    pair_timestamp_tokens = torch.zeros(2, dtype=torch.long)
    min_timestamp_tokens = torch.full((2,), 10, dtype=torch.long)
    position = torch.full((2,), 5, dtype=torch.long)
    sampling_seed = torch.full((), 124, dtype=torch.long)
    all_finished = torch.zeros((), dtype=torch.bool)
    sentence_end = torch.zeros(20, dtype=torch.bool)

    update_decode_control_(
        torch.tensor([4, 5]),
        torch.tensor([-0.1, -0.2]),
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
        sentence_end,
        eos_token_id=9,
        timestamp_begin=10,
        timestamps_after_sentence_end=True,
        stop_after_first_sentence=False,
    )

    assert generated_tokens.tolist() == [[4, -1, -1], [-1, -1, -1]]
    assert generated_lengths.tolist() == [1, 1]
    assert current_token.tolist() == [4, 9]
    assert finished.tolist() == [False, True]
    assert position.tolist() == [6, 6]
    assert sampling_seed.item() == 125
    assert not all_finished.item()
