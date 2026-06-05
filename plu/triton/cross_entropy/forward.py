from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from plu.ref.cross_entropy import cross_entropy as ref_cross_entropy
from plu.triton.cross_entropy.backward import cross_entropy_backward


@triton.jit
def _cross_entropy_forward_kernel(
    logits_ptr,
    labels_ptr,
    losses_ptr,
    vocab_size: tl.constexpr,
    ignore_index: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < vocab_size
    logits = tl.load(logits_ptr + row * vocab_size + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    label = tl.load(labels_ptr + row)
    valid = label != ignore_index

    row_max = tl.max(logits, axis=0)
    logsumexp = tl.log(tl.sum(tl.exp(logits - row_max), axis=0)) + row_max
    target = tl.load(logits_ptr + row * vocab_size + label, mask=valid, other=0.0).to(tl.float32)
    loss = logsumexp - target
    tl.store(losses_ptr + row, tl.where(valid, loss, 0.0))


class _TritonCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, labels: Tensor, ignore_index: int) -> Tensor:
        original_shape = logits.shape
        vocab_size = original_shape[-1]
        logits_2d = logits.contiguous().reshape(-1, vocab_size)
        labels_1d = labels.contiguous().reshape(-1)
        rows = logits_2d.shape[0]
        losses = torch.empty(rows, device=logits.device, dtype=torch.float32)

        if rows:
            block_size = triton.next_power_of_2(vocab_size)
            num_warps = 4 if block_size <= 2048 else 8
            _cross_entropy_forward_kernel[(rows,)](
                logits_2d,
                labels_1d,
                losses,
                vocab_size,
                ignore_index,
                block_size,
                num_warps=num_warps,
            )

        valid_count = labels_1d.ne(ignore_index).sum()
        loss = losses.sum() / valid_count.to(losses.dtype)
        ctx.save_for_backward(logits_2d, labels_1d, valid_count)
        ctx.ignore_index = ignore_index
        ctx.original_shape = original_shape
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        logits_2d, labels_1d, valid_count = ctx.saved_tensors
        dlogits = cross_entropy_backward(logits_2d, labels_1d, valid_count, grad_output, ctx.ignore_index)
        return dlogits.reshape(ctx.original_shape), None, None


def cross_entropy(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    if not logits.is_cuda:
        return ref_cross_entropy(logits, labels, ignore_index=ignore_index)
    return _TritonCrossEntropy.apply(logits, labels, ignore_index)

