from __future__ import annotations

from plu.triton.gelu_mlp.backward import linear_input_grad, linear_weight_bias_grad

__all__ = ["linear_input_grad", "linear_weight_bias_grad"]
