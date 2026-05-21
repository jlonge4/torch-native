"""
NKI kernel wrappers for Wan2.2 on Trainium 2.

Three kernels:
  - nki_rmsnorm    : drop-in for WanRMSNorm
  - nki_rope       : drop-in for rope_apply (eliminates @dynamo.disable + CPU round-trip)
  - nki_attn_cte   : drop-in for F.scaled_dot_product_attention (tiled flash attention)

Pattern (from torch_neuronx docs):
  1. wrap_nki + @nki.jit  — makes kernel traceable through torch.compile
  2. @nki_op              — registers wrapper as a PyTorch custom op
"""
import torch
from torch_neuronx import nki_op, wrap_nki
import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.embeddings.rope import RoPE as _nkilib_rope
from nkilib.core.attention.attention_cte import attention_cte as _nkilib_attn_cte

_wrapped_rope = wrap_nki(_nkilib_rope)
_wrapped_attn_cte = wrap_nki(_nkilib_attn_cte)


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────────────────

def _stream_shuffle_broadcast(src, dst):
    dst_npar = dst.shape[0]
    free_dim = dst.shape[1]
    shuffle_mask = [0] * 32
    assert dst_npar % 32 == 0
    for i in range(dst_npar // 32):
        nisa.nc_stream_shuffle(
            src=src[0:1, :],
            dst=dst[i * 32:(i + 1) * 32, 0:free_dim],
            shuffle_mask=shuffle_mask,
        )


@wrap_nki
@nki.jit()
def _nki_rmsnorm_kernel(input_tensor, weight, eps):
    MAX_P = 128
    output = nl.ndarray(input_tensor.shape, dtype=input_tensor.dtype, buffer=nl.shared_hbm)
    num_rows = input_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    num_chunks = (num_rows + MAX_P - 1) // MAX_P

    g_tile = nl.ndarray((1, hidden_size), dtype=weight.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_tile[0:1, 0:hidden_size],
                  src=weight.reshape((1, hidden_size))[0:1, 0:hidden_size])

    for i in nl.affine_range(num_chunks):
        p_start = i * MAX_P
        valid_rows = min(MAX_P, num_rows - p_start)

        a = nl.ndarray((MAX_P, hidden_size), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a[0:valid_rows, 0:hidden_size],
                      src=input_tensor[p_start:p_start + valid_rows, 0:hidden_size])

        t = nl.ndarray((MAX_P, hidden_size), dtype=input_tensor.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t, data1=a, data2=a, op=nl.multiply)

        sq_sum = nl.ndarray((MAX_P, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(dst=sq_sum, data=t, op=nl.add, axis=1)

        s = nl.ndarray((MAX_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=s, data=sq_sum, op0=nl.multiply,
                           operand0=1.0 / hidden_size, op1=nl.add, operand1=eps)
        nisa.activation(dst=s, data=s, op=nl.rsqrt)
        nisa.tensor_scalar(dst=t, data=a, operand0=s, op0=nl.multiply)

        g_bcast = nl.ndarray((MAX_P, hidden_size), dtype=g_tile.dtype, buffer=nl.sbuf)
        _stream_shuffle_broadcast(g_tile, g_bcast)
        nisa.tensor_tensor(dst=t, data1=t, data2=g_bcast, op=nl.multiply)
        nisa.dma_copy(dst=output[p_start:p_start + valid_rows, 0:hidden_size],
                      src=t[0:valid_rows, 0:hidden_size])

    return output


@nki_op("wan::nki_rmsnorm", mutates_args={})
def nki_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RMSNorm via NKI. x: [..., H], weight: [H]. Flattens leading dims internally."""
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1]).contiguous()
    out = _nki_rmsnorm_kernel(x_2d, weight.view(-1), 1e-5)
    return out.view(orig_shape)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────────────────────────────────────

@nki_op("wan::nki_rope", mutates_args={})
def nki_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE via nkilib.
    x:   [d_head, B, n_heads, S]
    cos: [d_head//2, B, S]
    sin: [d_head//2, B, S]
    Returns: [d_head, B, n_heads, S]
    """
    return _wrapped_rope(x, cos, sin)


# ─────────────────────────────────────────────────────────────────────────────
# Flash Attention (prefill / CTE)
# ─────────────────────────────────────────────────────────────────────────────

@nki_op("wan::nki_attn_cte", mutates_args={})
def nki_attn_cte(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Flash attention via nkilib attention_cte.
    q: [B*n_heads, S, d_head]  (pre-scaled)
    k: [B*n_heads, d_head, S]  (transposed)
    v: [B*n_heads, S, d_head]
    Returns: [B*n_heads, S, d_head]
    """
    return _wrapped_attn_cte(q, k, v, scale=1.0, causal_mask=False)
