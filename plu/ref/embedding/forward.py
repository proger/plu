from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def encoder_position_embedding(hidden_states: Tensor, position_weight: Tensor) -> Tensor:
    seq_len = hidden_states.shape[1]
    return hidden_states + position_weight[:seq_len].to(hidden_states.dtype)


def decoder_embedding(input_ids: Tensor, token_weight: Tensor, position_weight: Tensor, dtype: torch.dtype) -> Tensor:
    seq_len = input_ids.shape[1]
    token = F.embedding(input_ids, token_weight)
    position = position_weight[:seq_len]
    return (token + position).to(dtype)
