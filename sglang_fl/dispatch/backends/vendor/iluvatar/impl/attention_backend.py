# Iluvatar FA2 attention backend using Corex flash_attn.

from __future__ import annotations

import sys
import types

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

from sglang_fl.dispatch.backends.vendor.iluvatar.impl import flash_attn_ops


def _install_corex_flash_attn_shim() -> None:
    mod = types.ModuleType("sgl_kernel.flash_attn")
    mod.flash_attn_varlen_func = flash_attn_ops.flash_attn_varlen_func
    mod.flash_attn_with_kvcache = flash_attn_ops.flash_attn_with_kvcache
    mod.get_scheduler_metadata = flash_attn_ops.get_scheduler_metadata
    sys.modules["sgl_kernel.flash_attn"] = mod


def _patch_flashattention_backend_module() -> None:
    import sglang.srt.layers.attention.flashattention_backend as fab

    fab.flash_attn_varlen_func = flash_attn_ops.flash_attn_varlen_func
    fab.flash_attn_with_kvcache = flash_attn_ops.flash_attn_with_kvcache


class IluFlashAttentionBackend(FlashAttentionBackend):
    """FA2 backend for Iluvatar via Corex flash_attn."""

    def __init__(self, model_runner, **kwargs):
        _install_corex_flash_attn_shim()
        _patch_flashattention_backend_module()
        super().__init__(model_runner, fa_impl_ver=3, **kwargs)
        self.fa_impl_ver = 2
        self._get_scheduler_metadata = None
        self.flash_attn_varlen_func = flash_attn_ops.flash_attn_varlen_func
        self.flash_attn_with_kvcache = flash_attn_ops.flash_attn_with_kvcache
