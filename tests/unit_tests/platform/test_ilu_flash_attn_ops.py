import torch

from sglang_fl.dispatch.backends.vendor.iluvatar.impl import flash_attn_ops


def test_varlen_runs():
    H, D = 8, 64
    S = 128
    q = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
    cu = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    out = flash_attn_ops.flash_attn_varlen_func(
        q, k, v, cu, cu, S, S, causal=True
    )
    assert out.shape == (S, H, D)


def test_kvcache_paged_gather_runs():
    page_size = 16
    num_blocks = 32
    Hq, Hkv, D = 8, 2, 64
    B = 2
    q = torch.randn(B, 1, Hq, D, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(
        num_blocks, page_size, Hkv, D, device="cuda", dtype=torch.float16
    )
    v_cache = torch.randn(
        num_blocks, page_size, Hkv, D, device="cuda", dtype=torch.float16
    )
    cache_seqlens = torch.tensor([32, 48], dtype=torch.int32, device="cuda")
    page_table = torch.zeros(B, 4, dtype=torch.int32, device="cuda")
    for i, sl in enumerate([32, 48]):
        n_blocks = sl // page_size
        page_table[i, :n_blocks] = torch.arange(
            n_blocks, device="cuda", dtype=torch.int32
        ) + i * 8

    out = flash_attn_ops.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        causal=True,
    )
    assert out.shape == (B, 1, Hq, D)


def test_n_heads_k_page_size_one():
    k_paged = torch.randn(100, 1, 16, 64, device="cuda", dtype=torch.float16)
    assert flash_attn_ops._n_heads_k_from_cache(k_paged, gathered=False) == 16
    k_gathered = torch.randn(24, 16, 128, 64, device="cuda", dtype=torch.float16)
    assert flash_attn_ops._n_heads_k_from_cache(
        k_gathered, gathered=True, max_seqlen=128
    ) == 16
    k_capture = torch.randn(24, 1, 16, 64, device="cuda", dtype=torch.float16)
    assert flash_attn_ops._n_heads_k_from_cache(
        k_capture, gathered=True, max_seqlen=1
    ) == 16
    assert flash_attn_ops._n_heads_k_from_cache(
        k_capture, gathered=True, max_seqlen=0
    ) == 16


def test_gather_limits_by_cache_seqlens_not_page_table_width():
    page_size = 1
    num_blocks = 4096
    Hkv, D = 16, 128
    B = 24
    max_context = 8192
    k_cache = torch.randn(
        num_blocks, page_size, Hkv, D, device="cuda", dtype=torch.float16
    )
    v_cache = torch.randn(
        num_blocks, page_size, Hkv, D, device="cuda", dtype=torch.float16
    )
    cache_seqlens = torch.ones(B, dtype=torch.int32, device="cuda")
    page_table = torch.randint(
        0, num_blocks, (B, max_context), dtype=torch.int32, device="cuda"
    )
    k, v = flash_attn_ops._gather_paged_kv_cache(
        k_cache, v_cache, page_table, cache_seqlens
    )[:2]
    assert k.shape == (B, 1, Hkv, D)
    assert v.shape == (B, 1, Hkv, D)

    cache_seqlens_long = torch.full((B,), 48, dtype=torch.int32, device="cuda")
    page_table_long = torch.randint(
        0, num_blocks, (B, 48), dtype=torch.int32, device="cuda"
    )
    k_long, v_long = flash_attn_ops._gather_paged_kv_cache(
        k_cache, v_cache, page_table_long, cache_seqlens_long
    )[:2]
    assert k_long.shape == (B, Hkv, 48, D)
    assert v_long.shape == (B, Hkv, 48, D)


def test_extend_varlen_prefill_runs():
    page_size = 1
    num_blocks = 256
    Hq, Hkv, D = 16, 16, 128
    seq_lens = [64, 96]
    B = len(seq_lens)
    total_q = sum(seq_lens)
    q = torch.randn(total_q, Hq, D, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(
        num_blocks, page_size, Hkv, D, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.randn(
        num_blocks, page_size, Hkv, D, device="cuda", dtype=torch.bfloat16
    )
    cache_seqlens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    cu_q = torch.tensor(
        [0, seq_lens[0], total_q], dtype=torch.int32, device="cuda"
    )
    cu_k = cu_q.clone()
    page_table = torch.zeros(B, max(seq_lens), dtype=torch.int32, device="cuda")
    for i, sl in enumerate(seq_lens):
        page_table[i, :sl] = torch.arange(sl, device="cuda", dtype=torch.int32) + i * 128

    out = flash_attn_ops.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_q,
        cu_seqlens_k_new=cu_k,
        max_seqlen_q=max(seq_lens),
        causal=True,
    )
    assert out.shape == (total_q, Hq, D)


def test_packed_output_slices_query_heads_only():
    B, Hq, Hkv, D = 16, 16, 16, 128
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(64, 1, Hkv, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(64, 1, Hkv, D, device="cuda", dtype=torch.bfloat16)
    cache_seqlens = torch.ones(B, dtype=torch.int32, device="cuda")
    page_table = torch.zeros(B, 32, dtype=torch.int32, device="cuda")
    out = flash_attn_ops.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        causal=True,
    )
    assert out.shape == (B, 1, Hq, D)


def test_kvcache_decode_3d_query_runs():
    page_size = 1
    num_blocks = 64
    Hq, Hkv, D = 8, 2, 64
    B = 24
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(
        num_blocks, page_size, Hkv, D, device="cuda", dtype=torch.float16
    )
    v_cache = torch.randn(
        num_blocks, page_size, Hkv, D, device="cuda", dtype=torch.float16
    )
    cache_seqlens = torch.full((B,), 32, dtype=torch.int32, device="cuda")
    page_table = torch.arange(B * 32, device="cuda", dtype=torch.int32).view(B, 32)

    out = flash_attn_ops.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        causal=True,
    )
    assert out.shape == (B, 1, Hq, D)
