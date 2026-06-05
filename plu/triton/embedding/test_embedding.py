from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from plu.ref.embedding import decoder_embedding as ref_decoder_embedding
from plu.ref.embedding import encoder_position_embedding as ref_encoder_position_embedding
from plu.triton.embedding import decoder_embedding as triton_decoder_embedding
from plu.triton.embedding import encoder_position_embedding as triton_encoder_position_embedding


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def _clone_args(*args):
    return [arg.detach().clone().requires_grad_(arg.requires_grad) if torch.is_tensor(arg) else arg for arg in args]


def test_encoder_position_embedding_forward_backward():
    torch.manual_seed(0)
    hidden = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
    position = torch.randn(16, 8, device="cuda", requires_grad=True)
    triton_args = _clone_args(hidden, position)

    ref_out = ref_encoder_position_embedding(hidden, position)
    triton_out = triton_encoder_position_embedding(*triton_args)
    torch.testing.assert_close(triton_out, ref_out, atol=0, rtol=0)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    triton_out.backward(grad)
    for ref_arg, triton_arg in zip((hidden, position), triton_args):
        torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=0, rtol=0)


def test_decoder_embedding_forward_backward():
    torch.manual_seed(0)
    input_ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]], device="cuda")
    token_weight = torch.randn(20, 8, device="cuda", requires_grad=True)
    position_weight = torch.randn(16, 8, device="cuda", requires_grad=True)
    triton_args = _clone_args(token_weight, position_weight)

    ref_out = ref_decoder_embedding(input_ids, token_weight, position_weight, torch.float32)
    triton_out = triton_decoder_embedding(input_ids, *triton_args, torch.float32)
    torch.testing.assert_close(triton_out, ref_out, atol=0, rtol=0)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    triton_out.backward(grad)
    for ref_arg, triton_arg in zip((token_weight, position_weight), triton_args):
        torch.testing.assert_close(triton_arg.grad, ref_arg.grad, atol=0, rtol=0)
