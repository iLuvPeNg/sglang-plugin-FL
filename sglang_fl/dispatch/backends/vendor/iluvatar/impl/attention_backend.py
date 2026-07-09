"""Iluvatar Flash Attention backend (WIP).

Uses Corex ``flash_attn`` 2.8.3 (``flash_attn.flash_attn_interface``) instead of
sglang's ``jit_kernel.flash_attention`` / ``sgl_kernel``, which require SM>=80.

Implementation notes for the follow-up work:
  - prefill/extend: ``flash_attn_varlen_func`` (already verified on MR-V100)
  - decode/kvcache: ``flash_attn_with_kvcache`` with Corex layout
    ``k_cache: (batch, nheads_k, seqlen_cache, headdim)`` and ``is_qkv_packed=True``
  - CUDA graph: copy metadata hooks from ``TritonAttnBackend`` or
    ``FlashAttentionBackend`` (init_cuda_graph_state, capture/replay metadata)
  - Keep ``platform._ATTN_BACKEND_MAP["iluvatar"] = "triton"`` until this backend
    passes graph + accuracy tests; then switch default to ``iluvatar_fa``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class IluvatarFlashAttentionBackend(AttentionBackend):
    """OOT FA backend for Iluvatar; registered as ``iluvatar_fa``."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        raise NotImplementedError(
            "IluvatarFlashAttentionBackend is scaffold-only on feat/iluvatar-flash-attention"
        )
