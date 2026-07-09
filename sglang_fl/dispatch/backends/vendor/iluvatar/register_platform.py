# Iluvatar OOT attention backend registration.

import logging

from sglang.srt.layers.attention.attention_registry import register_attention_backend
from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES, add_attention_backend_choices

logger = logging.getLogger(__name__)


@register_attention_backend("fa2")
def _create_ilu_fa2_backend(runner):
    from sglang_fl.dispatch.backends.vendor.iluvatar.impl.attention_backend import (
        IluFlashAttentionBackend,
    )

    return IluFlashAttentionBackend(runner)


def ensure_fa2_cli_choice() -> None:
    """Expose ``fa2`` to ``--attention-backend`` argparse choices."""
    if "fa2" not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices(["fa2"])


ensure_fa2_cli_choice()
