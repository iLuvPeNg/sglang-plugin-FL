"""Iluvatar OOT attention backend registration.

Wired by ``PlatformFL.init_backend()`` on worker startup. Registers
``iluvatar_fa`` into sglang's ``ATTENTION_BACKENDS`` without modifying
sglang-preview. Enable via::

    # after implementation is complete:
    # _ATTN_BACKEND_MAP["iluvatar"] = "iluvatar_fa"
    # or: --attention-backend iluvatar_fa
"""

import logging

from sglang.srt.layers.attention.attention_registry import register_attention_backend

logger = logging.getLogger(__name__)


@register_attention_backend("iluvatar_fa")
def _create_iluvatar_fa_backend(runner):
    from sglang_fl.dispatch.backends.vendor.iluvatar.impl.attention_backend import (
        IluvatarFlashAttentionBackend,
    )

    return IluvatarFlashAttentionBackend(runner)
