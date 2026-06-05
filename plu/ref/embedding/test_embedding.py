from __future__ import annotations

import torch
import torch.nn.functional as F

from plu.ref.embedding import decoder_embedding, encoder_position_embedding


def test_encoder_position_embedding_matches_add():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 4, 8)
    position_weight = torch.randn(16, 8)
    torch.testing.assert_close(encoder_position_embedding(hidden_states, position_weight), hidden_states + position_weight[:4])


def test_decoder_embedding_matches_token_plus_position():
    torch.manual_seed(0)
    input_ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
    token_weight = torch.randn(20, 8)
    position_weight = torch.randn(16, 8)
    expected = F.embedding(input_ids, token_weight) + position_weight[:4]
    torch.testing.assert_close(decoder_embedding(input_ids, token_weight, position_weight, torch.float32), expected)
