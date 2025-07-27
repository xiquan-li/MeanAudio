from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from meanaudio.ext.rotary_embeddings import apply_rope
from meanaudio.model.low_level import MLP, ChannelLastConv1d, ConvMLP


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift  # scale is actually the add term for x (res connect for modulation)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    # flash attention is not compatible with JVP calculation
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        out = F.scaled_dot_product_attention(q, k, v)
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    return out


class SelfAttention(nn.Module):

    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.RMSNorm(dim // nheads)
        self.k_norm = nn.RMSNorm(dim // nheads)

        self.split_into_heads = Rearrange('b n (h d j) -> b h n d j',
                                          h=nheads,
                                          d=dim // nheads,
                                          j=3)

    def pre_attention(  # get qkv for input x, apply rotary pos embedding if needed
            self, x: torch.Tensor,
            rot: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch_size * n_tokens * n_channels
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)  # chunk: split the input into 3 components 
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)

        return q, k, v

    def forward(
            self,
            x: torch.Tensor,  # batch_size * n_tokens * n_channels
    ) -> torch.Tensor:
        q, v, k = self.pre_attention(x)
        out = attention(q, k, v)  
        return out


class MMDitSingleBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 nhead: int,
                 mlp_ratio: float = 4.0,
                 pre_only: bool = False,
                 kernel_size: int = 7,
                 padding: int = 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, nhead)

        self.pre_only = pre_only
        if pre_only:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        else:
            if kernel_size == 1:
                self.linear1 = nn.Linear(dim, dim)
            else:
                self.linear1 = ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

            if kernel_size == 1:
                self.ffn = MLP(dim, int(dim * mlp_ratio))
            else:
                self.ffn = ConvMLP(dim,
                                   int(dim * mlp_ratio),
                                   kernel_size=kernel_size,
                                   padding=padding)

            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: Optional[torch.Tensor]):
        """get qkv from x and modulation coefficients from condition"""
        # x: BS * N * D
        # cond: BS * D
        modulation = self.adaLN_modulation(c)  # get modulation coefficients 
        if self.pre_only:
            (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
             gate_mlp) = modulation.chunk(6, dim=-1)

        x = modulate(self.norm1(x), shift_msa, scale_msa)  # first AdaLN
        q, k, v = self.attn.pre_attention(x, rot)  # linear for qkv
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor]):
        if self.pre_only:
            return x

        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa  # first linear/ConvMLP & scaling & residual
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)  # second AdaLN
        x = x + self.ffn(r) * gate_mlp  # second linear/ConvMLP & scaling & residual 

        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                rot: Optional[torch.Tensor]) -> torch.Tensor:
        # x: BS * N * D
        # cond: BS * D
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        attn_out = attention(*x_qkv)
        x = self.post_attention(x, attn_out, x_conditions)

        return x


class JointBlock(nn.Module):

    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, pre_only: bool = False):
        super().__init__()
        self.pre_only = pre_only
        self.latent_block = MMDitSingleBlock(dim,
                                             nhead,
                                             mlp_ratio,
                                             pre_only=False,
                                             kernel_size=3,
                                             padding=1)
        self.text_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)

    def forward(self, latent: torch.Tensor, text_f: torch.Tensor,
                global_c: torch.Tensor, extended_c: torch.Tensor, 
                latent_rot: torch.Tensor, text_rot: torch.Tensor, 
                ) -> tuple[torch.Tensor, torch.Tensor]:  
        # latent: BS * N1 * D
        # c: BS * (1/N) * D
        x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, rot=latent_rot)  # fine-grained features are only used for the audio branch
        t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=text_rot)  

        latent_len = latent.shape[1]
        text_len = text_f.shape[1]

        joint_qkv = [torch.cat([x_qkv[i], t_qkv[i]], dim=2) for i in range(3)]

        attn_out = attention(*joint_qkv)  # core of joint block: joint attention
        x_attn_out = attn_out[:, :latent_len]  
        t_attn_out = attn_out[:, latent_len:]

        latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
        if not self.pre_only:
            text_f = self.text_block.post_attention(text_f, t_attn_out, t_mod)  # for pre-only layer we don't do post attention for condition features

        return latent, text_f


class FinalBlock(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent
