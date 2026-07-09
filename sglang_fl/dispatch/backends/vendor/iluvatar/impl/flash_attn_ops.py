# Iluvatar flash attention wrappers using Corex flash_attn.

from __future__ import annotations

from typing import Any, Optional

import torch

from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func as _corex_flash_attn_varlen_func,
)
from flash_attn.flash_attn_interface import (
    flash_attn_with_kvcache as _corex_flash_attn_with_kvcache,
)
from flash_attn.flash_attn_interface import maybe_contiguous

_VARLEN_ALLOWED = frozenset(
    {
        "dropout_p",
        "softmax_scale",
        "causal",
        "window_size",
        "softcap",
        "alibi_slopes",
        "deterministic",
        "return_attn_probs",
        "block_table",
        "use_alibi",
        "alibi_mode",
        "imp_mode",
        "out",
        "bias",
    }
)

_KVCACHE_ALLOWED = frozenset(
    {
        "k",
        "v",
        "rotary_cos",
        "rotary_sin",
        "cache_seqlens",
        "cache_batch_idx",
        "cache_leftpad",
        "block_table",
        "softmax_scale",
        "causal",
        "window_size",
        "softcap",
        "rotary_interleaved",
        "alibi_slopes",
        "num_splits",
        "return_softmax_lse",
        "alibi_mode",
        "is_qkv_packed",
    }
)

_STRIPPED_KWARGS = frozenset(
    {
        "page_table",
        "cu_seqlens_q",
        "cu_seqlens_k_new",
        "max_seqlen_q",
        "k_descale",
        "v_descale",
        "scheduler_metadata",
        "pack_gqa",
        "sm_margin",
        "ver",
        "attention_chunk",
        "qv",
        "rotary_seqlens",
        "q_descale",
        "sinks",
        "score_mod",
        "aux_tensors",
    }
)


def get_scheduler_metadata(*_args, **_kwargs):
    return None


def _filter_kwargs(kwargs: dict[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
    out = {}
    for key, value in kwargs.items():
        if key in _STRIPPED_KWARGS:
            continue
        if key in allowed:
            out[key] = value
    return out


def _gather_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _num_blocks, page_size, _n_heads, _head_dim = k_cache.shape
    batch_size = page_table.shape[0]
    device = k_cache.device

    k_src = k_cache.contiguous()
    v_src = v_cache.contiguous()
    page_table = page_table.contiguous()
    cache_seqlens = cache_seqlens.contiguous()

    max_len = int(cache_seqlens.max().item())
    table_cap = page_table.shape[1] * page_size
    max_len = min(max_len, table_cap)
    if max_len <= 0:
        max_len = 1

    positions = torch.arange(max_len, device=device, dtype=torch.int32)
    block_indices = positions // page_size
    offsets = positions % page_size

    blocks = page_table[:, block_indices.long()]
    offs = offsets.unsqueeze(0).expand(batch_size, -1)
    k_tokens = k_src[blocks, offs]
    v_tokens = v_src[blocks, offs]

    if max_len == 1:
        return k_tokens.contiguous(), v_tokens.contiguous()
    k_cont = k_tokens.permute(0, 2, 1, 3).contiguous()
    v_cont = v_tokens.permute(0, 2, 1, 3).contiguous()
    return k_cont, v_cont


def _n_heads_k_from_cache(
    k_cache: torch.Tensor, *, gathered: bool, max_seqlen: int = 0
) -> int:
    if k_cache.ndim != 4:
        raise ValueError(f"unexpected k_cache shape: {k_cache.shape}")
    if gathered:
        return int(k_cache.shape[2] if max_seqlen == 1 else k_cache.shape[1])
    return int(k_cache.shape[2])


def _ensure_bshd(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"{name} must be 3D or 4D, got shape {tuple(x.shape)}")


def _n_heads_q_from_q(q: torch.Tensor) -> int:
    if q.dim() == 3:
        return int(q.shape[1])
    if q.dim() == 4:
        return int(q.shape[2])
    raise ValueError(f"q must be 3D or 4D, got shape {tuple(q.shape)}")


def _slice_corex_packed_output(result, n_heads_q: int):
    if isinstance(result, tuple):
        out = result[0][:, :, :n_heads_q, :].contiguous()
        return (out, *result[1:])
    return result[:, :, :n_heads_q, :].contiguous()


def _pack_q_for_corex(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    *,
    n_heads_k: int,
) -> torch.Tensor:
    q = _ensure_bshd(q, "q")
    if k is not None and v is not None:
        k = _ensure_bshd(k, "k")
        v = _ensure_bshd(v, "v")
        return torch.cat([q, k, v], dim=2)
    batch, seqlen, _n_heads_q, head_dim = q.shape
    zeros = q.new_zeros(batch, seqlen, n_heads_k, head_dim)
    return torch.cat([q, zeros, zeros], dim=2)


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    **kwargs,
):
    if kwargs.get("page_table") is not None and kwargs.get("block_table") is None:
        kwargs["block_table"] = kwargs.pop("page_table")
    else:
        kwargs.pop("page_table", None)

    if kwargs.pop("return_softmax_lse", False):
        kwargs["return_attn_probs"] = True

    filtered = _filter_kwargs(kwargs, _VARLEN_ALLOWED)
    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)
    return _corex_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        **filtered,
    )


def _gathered_cache_to_shd(k_gathered: torch.Tensor, max_seqlen: int) -> torch.Tensor:
    if max_seqlen == 1:
        return k_gathered[0].contiguous()
    return k_gathered[0].permute(1, 0, 2).contiguous()


def _flash_attn_varlen_from_paged_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    **kwargs,
):
    batch_size = page_table.shape[0]
    k_parts = []
    v_parts = []
    for i in range(batch_size):
        cs_i = cache_seqlens[i : i + 1].contiguous()
        max_i = int(cs_i.item())
        k_i, v_i = _gather_paged_kv_cache(
            k_cache, v_cache, page_table[i : i + 1], cs_i
        )
        k_parts.append(_gathered_cache_to_shd(k_i, max_i))
        v_parts.append(_gathered_cache_to_shd(v_i, max_i))

    k = torch.cat(k_parts, dim=0)
    v = torch.cat(v_parts, dim=0)
    if cu_seqlens_k is None:
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens.to(torch.int32), dim=0),
            (1, 0),
        )
    max_seqlen_k = int(
        (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
    )
    return flash_attn_varlen_func(
        maybe_contiguous(q),
        maybe_contiguous(k),
        maybe_contiguous(v),
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        **kwargs,
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    **kwargs,
):
    page_table = kwargs.pop("page_table", None)
    cu_seqlens_q = kwargs.pop("cu_seqlens_q", None)
    cu_seqlens_k = kwargs.pop("cu_seqlens_k_new", None)
    max_seqlen_q = kwargs.pop("max_seqlen_q", None)
    filtered = _filter_kwargs(kwargs, _KVCACHE_ALLOWED)
    gathered = False
    max_seqlen = 0
    n_heads_q = _n_heads_q_from_q(q)

    if page_table is not None:
        cache_seqlens = filtered.get("cache_seqlens")
        if cache_seqlens is None:
            raise ValueError("cache_seqlens is required when page_table is set")
        if not isinstance(cache_seqlens, torch.Tensor):
            cache_seqlens = torch.full(
                (page_table.shape[0],),
                int(cache_seqlens),
                dtype=torch.int32,
                device=page_table.device,
            )
        cache_seqlens = cache_seqlens.contiguous()

        if (
            cu_seqlens_q is not None
            and q.dim() == 3
            and q.shape[0] != page_table.shape[0]
        ):
            if max_seqlen_q is None:
                max_seqlen_q = int(
                    (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
                )
            varlen_kwargs = {
                k: v for k, v in filtered.items() if k != "cache_seqlens"
            }
            return _flash_attn_varlen_from_paged_cache(
                q,
                k_cache,
                v_cache,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                **varlen_kwargs,
            )

        max_seqlen = int(cache_seqlens.max().item())
        k_cache, v_cache = _gather_paged_kv_cache(
            k_cache, v_cache, page_table, cache_seqlens
        )
        gathered = True
        filtered["cache_seqlens"] = cache_seqlens
    else:
        max_seqlen = 0

    q = maybe_contiguous(q)
    if k is not None:
        k = maybe_contiguous(k)
    if v is not None:
        v = maybe_contiguous(v)
    q = _pack_q_for_corex(
        q,
        k,
        v,
        n_heads_k=_n_heads_k_from_cache(
            k_cache, gathered=gathered, max_seqlen=max_seqlen
        ),
    )
    filtered.pop("k", None)
    filtered.pop("v", None)
    filtered["is_qkv_packed"] = True

    q = maybe_contiguous(q)
    k_cache = maybe_contiguous(k_cache)
    v_cache = maybe_contiguous(v_cache)

    result = _corex_flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        **filtered,
    )
    return _slice_corex_packed_output(result, n_heads_q)
