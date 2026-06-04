# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Tuple

from torch import Tensor, nn
import torch
import torch.nn.functional as F

from .utils import cat_keep_shapes, uncat_with_shapes

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        use_qk_norm: bool = False,
        use_sparse_index: bool = False,
        use_triton_sparse: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.use_triton_sparse = use_triton_sparse
        # VGGT-Omega change: the aggregator checkpoint was trained with Q/K
        # normalization, while upstream DINOv3 attention does not expose it.
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(head_dim, eps=1e-5)
        else:
            self.q_norm = None
            self.k_norm = None

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_sparse_index:
            self.dummy_k = nn.Parameter(torch.randn(1, 1, dim))
            nn.init.normal_(self.dummy_k, std=1e-6)
            self.dummy_v = nn.Parameter(torch.randn(1, 1, dim))
            nn.init.normal_(self.dummy_v, std=1e-6)
        else:
            self.dummy_k = None
            self.dummy_v = None

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None, index=None, ps_idx: int = 0) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope, index=index, ps_idx=ps_idx)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None, index=None, ps_idx: int = 0) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        if index is not None:
            return self._compute_sparse_attention(q, k, v, index, ps_idx, B, N, C)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])

    def _compute_sparse_attention(
        self, q: Tensor, k: Tensor, v: Tensor,
        index: Tensor, ps_idx: int, B: int, N: int, C: int,
    ) -> Tensor:
        """Sparse indexed attention for global inter-frame blocks.

        Special tokens (first ps_idx per frame) attend to all tokens via full
        attention.  Patch tokens attend only to the K tokens selected by index.

        index: (B, V*patches, K), 1-indexed — position 0 is a learned dummy token.
        """
        head_dim = C // self.num_heads
        scale = head_dim ** -0.5

        # V * (ps_idx + patches) == N,  V * patches == index.shape[1]
        V_frames = (N - index.shape[1]) // ps_idx

        q = q * scale  # [B, num_heads, N, head_dim]

        # Split into special-token queries and patch queries
        q = q.view(B, self.num_heads, V_frames, N // V_frames, head_dim)
        q_A = q[:, :, :, :ps_idx, :].reshape(B, self.num_heads, V_frames * ps_idx, head_dim)
        q_B = q[:, :, :, ps_idx:, :].reshape(B, self.num_heads, index.shape[1], head_dim)

        # Special tokens: full attention over all N tokens
        attn_A = q_A @ k.transpose(-2, -1)   # [B, num_heads, V*ps_idx, N]
        attn_A = attn_A.softmax(dim=-1)
        x_A = attn_A @ v  # [B, num_heads, V*ps_idx, head_dim]

        # Build key/value pool: dummy token at position 0, all N tokens after
        k_flat = k.permute(0, 2, 1, 3).reshape(B, N, C)   # [B, N, C]
        v_flat = v.permute(0, 2, 1, 3).reshape(B, N, C)
        k_pool = torch.cat([self.dummy_k.expand(B, -1, -1), k_flat], dim=1)  # [B, N+1, C]
        v_pool = torch.cat([self.dummy_v.expand(B, -1, -1), v_flat], dim=1)

        # Sparse attention for patch tokens — PyTorch gather path or Triton kernel
        if self.use_triton_sparse:
            # Triton path: reshape pool to [B, num_heads, N+1, head_dim] and pass
            # unscaled q_B (Triton kernel applies its own 1/sqrt(head_dim) scaling)
            k_pool_t = k_pool.view(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v_pool_t = v_pool.view(B, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
            x_B = mha_indexed_attention(q_B / scale, k_pool_t, v_pool_t, index)
        else:
            # Gather K keys/values for each patch query
            idx_flat = index.reshape(B, -1)  # [B, V*patches*K]
            k_sel = k_pool.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, C))
            k_sel = k_sel.view(B, index.shape[1], index.shape[2], self.num_heads, head_dim)
            k_sel = k_sel.permute(0, 3, 1, 2, 4)  # [B, num_heads, V*patches, K, head_dim]
            v_sel = v_pool.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, C))
            v_sel = v_sel.view(B, index.shape[1], index.shape[2], self.num_heads, head_dim)
            v_sel = v_sel.permute(0, 3, 1, 2, 4)  # [B, num_heads, V*patches, K, head_dim]
            attn_B = torch.einsum('bhmc,bhmkc->bhmk', q_B, k_sel)   # [B, num_heads, V*patches, K]
            attn_B = attn_B.softmax(dim=-1)
            x_B = torch.einsum('bhmk,bhmkc->bhmc', attn_B, v_sel)   # [B, num_heads, V*patches, head_dim]

        # Reassemble in original per-frame order: [ps_idx tokens, patches]
        x_A = x_A.view(B, self.num_heads, V_frames, ps_idx, head_dim)
        x_B = x_B.view(B, self.num_heads, V_frames, -1, head_dim)
        x = torch.cat([x_A, x_B], dim=3)       # [B, num_heads, V, T, head_dim]
        x = x.reshape(B, self.num_heads, N, head_dim)

        return x.transpose(1, 2).reshape(B, N, C)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ---------------------------------------------------------------------------
# Triton sparse indexed attention
# In testing, the Triton path is slower than the PyTorch gather path with
# similar GPU memory usage.  It is provided as an optional backend.
# ---------------------------------------------------------------------------

class _IndexedAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, idx):
        B, H, M, C = Q.shape
        K_size = idx.shape[-1]

        out = torch.empty_like(Q)
        probs = torch.empty((B, H, M, K_size), device=Q.device, dtype=Q.dtype)

        grid = (B * H * M,)
        _triton_indexed_attn_fwd[grid](
            Q, K, V, idx,
            probs, out,
            B, H, M, K.shape[2], C, K_size,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *idx.stride(),
            *probs.stride(),
            *out.stride(),
            BLOCK_C=1024,
        )

        ctx.save_for_backward(Q, K, V, idx, probs)
        return out

    @staticmethod
    def backward(ctx, dout):
        Q, K, V, idx, probs = ctx.saved_tensors

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        B, H, M, C = Q.shape
        K_size = idx.shape[-1]

        grid = (B * H * M,)
        _triton_indexed_attn_bwd[grid](
            Q, K, V, idx,
            probs, dout,
            dQ, dK, dV,
            B, H, M, K.shape[2], C, K_size,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *idx.stride(),
            *probs.stride(),
            *dout.stride(),
            BLOCK_C=1024,
        )

        return dQ, dK, dV, None


def mha_indexed_attention(Q, K, V, idx):
    """Triton-accelerated sparse indexed multi-head attention.

    Q:   [B, num_heads, M, head_dim]  — unscaled queries
    K:   [B, num_heads, N, head_dim]  — full key pool (including dummy at 0)
    V:   [B, num_heads, N, head_dim]  — full value pool
    idx: [B, M, K_size]              — 1-indexed positions to attend to
    """
    return _IndexedAttentionFn.apply(Q, K, V, idx)


if TRITON_AVAILABLE:
    @triton.jit
    def _triton_indexed_attn_fwd(
        Q_ptr, K_ptr, V_ptr, IDX_ptr,
        PROBS_ptr, OUT_ptr,
        B, H, M, N, C, K_size,
        stride_qb, stride_qh, stride_qm, stride_qc,
        stride_kb, stride_kh, stride_kn, stride_kc,
        stride_vb, stride_vh, stride_vn, stride_vc,
        stride_ib, stride_im, stride_ik,
        stride_pb, stride_ph, stride_pm, stride_pk,
        stride_ob, stride_oh, stride_om, stride_oc,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)

        m = pid % M
        h = (pid // M) % H
        b = pid // (M * H)

        q_ptr = Q_ptr + b * stride_qb + h * stride_qh + m * stride_qm
        idx_ptr = IDX_ptr + b * stride_ib + m * stride_im
        probs_ptr = PROBS_ptr + b * stride_pb + h * stride_ph + m * stride_pm
        out_ptr = OUT_ptr + b * stride_ob + h * stride_oh + m * stride_om

        inv_sqrt_c = 1.0 / tl.sqrt(C.to(tl.float32))
        neg_inf = -1e30

        max_score = neg_inf
        for i in range(K_size):
            n_idx = tl.load(idx_ptr + i * stride_ik).to(tl.int32)

            score = 0.0
            for c_start in range(0, C, BLOCK_C):
                offs = c_start + tl.arange(0, BLOCK_C)
                mask = offs < C
                q = tl.load(q_ptr + offs * stride_qc, mask=mask, other=0.0)
                k_ptr_ = K_ptr + b * stride_kb + h * stride_kh + n_idx * stride_kn
                k = tl.load(k_ptr_ + offs * stride_kc, mask=mask, other=0.0)
                score += tl.sum(q * k, 0)

            score *= inv_sqrt_c
            tl.store(probs_ptr + i * stride_pk, score)
            max_score = tl.maximum(max_score, score)

        sum_exp = 0.0
        for i in range(K_size):
            s = tl.load(probs_ptr + i * stride_pk)
            e = tl.exp(s - max_score)
            tl.store(probs_ptr + i * stride_pk, e)
            sum_exp += e

        inv_sum = 1.0 / sum_exp

        for c_start in range(0, C, BLOCK_C):
            offs = c_start + tl.arange(0, BLOCK_C)
            mask = offs < C
            acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
            for i in range(K_size):
                p = tl.load(probs_ptr + i * stride_pk) * inv_sum
                tl.store(probs_ptr + i * stride_pk, p)
                n_idx = tl.load(idx_ptr + i * stride_ik).to(tl.int32)
                v_ptr_ = V_ptr + b * stride_vb + h * stride_vh + n_idx * stride_vn
                v = tl.load(v_ptr_ + offs * stride_vc, mask=mask, other=0.0)
                acc += p * v
            tl.store(out_ptr + offs * stride_oc, acc, mask=mask)

    @triton.jit
    def _triton_indexed_attn_bwd(
        Q_ptr, K_ptr, V_ptr, IDX_ptr,
        PROBS_ptr, DOUT_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        B, H, M, N, C, K_size,
        stride_qb, stride_qh, stride_qm, stride_qc,
        stride_kb, stride_kh, stride_kn, stride_kc,
        stride_vb, stride_vh, stride_vn, stride_vc,
        stride_ib, stride_im, stride_ik,
        stride_pb, stride_ph, stride_pm, stride_pk,
        stride_db, stride_dh, stride_dm, stride_dc,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)

        m = pid % M
        h = (pid // M) % H
        b = pid // (M * H)

        q_ptr = Q_ptr + b * stride_qb + h * stride_qh + m * stride_qm
        idx_ptr = IDX_ptr + b * stride_ib + m * stride_im
        probs_ptr = PROBS_ptr + b * stride_pb + h * stride_ph + m * stride_pm
        dout_ptr = DOUT_ptr + b * stride_db + h * stride_dh + m * stride_dm
        dQ_ptr_ = dQ_ptr + b * stride_qb + h * stride_qh + m * stride_qm

        inv_sqrt_c = 1.0 / tl.sqrt(C.to(tl.float32))

        q = tl.load(q_ptr + tl.arange(0, BLOCK_C) * stride_qc, mask=tl.arange(0, BLOCK_C) < C, other=0.0)
        dout = tl.load(dout_ptr + tl.arange(0, BLOCK_C) * stride_dc, mask=tl.arange(0, BLOCK_C) < C, other=0.0)

        o = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for i in range(K_size):
            p = tl.load(probs_ptr + i * stride_pk)
            n_idx = tl.load(idx_ptr + i * stride_ik).to(tl.int32)
            v_ptr_ = V_ptr + b * stride_vb + h * stride_vh + n_idx * stride_vn
            v = tl.load(v_ptr_ + tl.arange(0, BLOCK_C) * stride_vc, mask=tl.arange(0, BLOCK_C) < C, other=0.0)
            o += p * v

        dQ = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for i in range(K_size):
            p = tl.load(probs_ptr + i * stride_pk)
            n_idx = tl.load(idx_ptr + i * stride_ik).to(tl.int32)

            k_ptr_ = K_ptr + b * stride_kb + h * stride_kh + n_idx * stride_kn
            v_ptr_ = V_ptr + b * stride_vb + h * stride_vh + n_idx * stride_vn
            k = tl.load(k_ptr_ + tl.arange(0, BLOCK_C) * stride_kc, mask=tl.arange(0, BLOCK_C) < C, other=0.0)
            v = tl.load(v_ptr_ + tl.arange(0, BLOCK_C) * stride_vc, mask=tl.arange(0, BLOCK_C) < C, other=0.0)

            delta = tl.sum(dout * (v - o), 0)
            ds = p * delta

            dQ += ds * k * inv_sqrt_c

            tl.atomic_add(
                dV_ptr + b * stride_vb + h * stride_vh + n_idx * stride_vn +
                tl.arange(0, BLOCK_C) * stride_vc,
                p * dout,
                mask=tl.arange(0, BLOCK_C) < C,
            )
            tl.atomic_add(
                dK_ptr + b * stride_kb + h * stride_kh + n_idx * stride_kn +
                tl.arange(0, BLOCK_C) * stride_kc,
                ds * q * inv_sqrt_c,
                mask=tl.arange(0, BLOCK_C) < C,
            )

        tl.store(dQ_ptr_ + tl.arange(0, BLOCK_C) * stride_qc, dQ, mask=tl.arange(0, BLOCK_C) < C)
else:
    # Stubs so the rest of the module can reference the names regardless
    def _triton_indexed_attn_fwd(*args, **kwargs):
        raise RuntimeError("Triton is not installed; cannot use use_triton_sparse=True")

    def _triton_indexed_attn_bwd(*args, **kwargs):
        raise RuntimeError("Triton is not installed; cannot use use_triton_sparse=True")
