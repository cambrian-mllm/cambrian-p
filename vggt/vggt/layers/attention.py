# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


import logging
import os
import warnings
from typing import Union, Tuple

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.is_causal = is_causal

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None,
                past_key_values=None, use_cache=False) -> Union[Tensor, Tuple[Tensor, Tuple]]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # --- KV cache: concat past keys/values ---
        pos_k = pos
        if use_cache:
            k = k.unsqueeze(2)
            v = v.unsqueeze(2)
            if past_key_values is not None:
                past_k, past_v = past_key_values
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_kv = (k, v)
            a, b, c, d, e = k.shape
            k = k.reshape(a, b, c * d, e)
            v = v.reshape(a, b, c * d, e)
            if pos_k is not None:
                pos_k = pos_k.repeat(1, c, 1)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos_k)

        if self.fused_attn:
            if use_cache:
                # Cache is inherently causal — no mask needed
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=self.is_causal,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if self.is_causal and not use_cache:
                N_k = k.shape[-2]
                causal_mask = torch.triu(torch.ones(N, N_k, dtype=torch.bool, device=attn.device), diagonal=1)
                attn = attn.masked_fill(causal_mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if use_cache:
            return x, new_kv
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x