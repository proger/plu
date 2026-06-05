from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def lora_linear(
    x: Tensor,
    adapter_input: Tensor,
    base_weight: Tensor,
    base_bias: Tensor | None,
    lora_a_weight: Tensor,
    lora_b_weight: Tensor,
    scaling: float,
) -> Tensor:
    base_bias = None if base_bias is None else base_bias.to(x.dtype)
    result = F.linear(x, base_weight.to(x.dtype), base_bias)
    hidden = F.linear(adapter_input, lora_a_weight.to(adapter_input.dtype))
    update = F.linear(hidden, lora_b_weight.to(hidden.dtype)) * scaling
    return result + update.to(result.dtype)
