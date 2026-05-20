# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

from ..modules.model import sinusoidal_embedding_1d
from .ulysses import distributed_attention
from .util import gather_forward, get_rank, get_world_size


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@torch.amp.autocast('cpu', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2, 2]  — (cos, sin) stacked, float32, no complex dtypes.
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    # split freqs along the c axis for temporal / height / width components
    fs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    sp_size = get_world_size()
    sp_rank = get_rank()

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # build per-token cos/sin: [seq_len, 1, c]
        cos_i = torch.cat([
            fs[0][:f, :, 0].view(f, 1, 1, -1).expand(f, h, w, -1),
            fs[1][:h, :, 0].view(1, h, 1, -1).expand(f, h, w, -1),
            fs[2][:w, :, 0].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)

        sin_i = torch.cat([
            fs[0][:f, :, 1].view(f, 1, 1, -1).expand(f, h, w, -1),
            fs[1][:h, :, 1].view(1, h, 1, -1).expand(f, h, w, -1),
            fs[2][:w, :, 1].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)

        # pad to full sequence length across all SP ranks, then slice this rank's shard
        cos_i = pad_freqs(cos_i, s * sp_size)
        sin_i = pad_freqs(sin_i, s * sp_size)
        cos_i = cos_i[sp_rank * s:(sp_rank + 1) * s]
        sin_i = sin_i[sp_rank * s:(sp_rank + 1) * s]

        # real-valued 2D rotation — no view_as_complex, no float64
        xi = x[i, :s].float().reshape(s, n, c, 2)
        x0, x1 = xi[..., 0], xi[..., 1]
        out = torch.stack([x0 * cos_i - x1 * sin_i,
                           x0 * sin_i + x1 * cos_i], dim=-1).flatten(2)

        tail = x[i, s:]
        if tail.shape[0] > 0:
            out = torch.cat([out.to(x.dtype), tail])
        else:
            out = out.to(x.dtype)

        output.append(out)
    return torch.stack(output)


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cpu', enabled=False):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel — shard sequence across ranks
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # gather sequence from all ranks
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
    )

    x = x.flatten(2)
    x = self.o(x)
    return x
