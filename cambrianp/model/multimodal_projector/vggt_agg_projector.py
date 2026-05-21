import torch
import torch.nn as nn
from typing import Callable

from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter


class VGGTAggProjector(nn.Module):
    def __init__(self, input_dim, dim=896, num_heads=16, mlp_ratio=2.0, depth=1, qkv_bias=True, proj_bias=True, 
                 ffn_bias=True, qk_norm=True, rope_freq=100, init_values=0.01, patch_size=14,
                 patch_start_idx=1):
        super().__init__()
        
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.patch_size = patch_size
        self.patch_start_idx = patch_start_idx
        
        if input_dim != dim:
            self.input_proj = nn.Linear(input_dim, dim)
        else:
            self.input_proj = nn.Identity()
        
        self.frame_blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=self.rope,
            )
        ] * depth)
        
        self.global_blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=self.rope,
            )
        ] * depth)
        
        self.depth = depth
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        B, S, P, C = x.shape
        
        
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, self.patch_size, self.patch_size, device=x.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(x.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        frame_intermediates = []
        global_intermediates = []
        for i in range(self.depth):
            x = self._forward_frame_blocks(x, pos, i)
            frame_intermediates.append(x)
            x = self._forward_global_blocks(x, pos, i)
            global_intermediates.append(x)
        
        final_x = torch.cat([frame_intermediates[-1], global_intermediates[-1]], dim=-1)
        
        return final_x

    def _forward_frame_blocks(self, x, pos, i):
        B, S, P, C = x.shape
        
        if x.shape != (B * S, P, C):
            x = x.view(B, S, P, C).view(B * S, P, C)
        
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)
        
        x = self.frame_blocks[i](x, pos=pos)
        
        return x.view(B, S, P, C)
        
        
    def _forward_global_blocks(self, x, pos, i):
        B, S, P, C = x.shape
        
        if x.shape != (B, S * P, C):
            x = x.view(B, S, P, C).view(B, S * P, C)
        
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)
        
        x = self.global_blocks[i](x, pos=pos)

        return x.view(B, S, P, C)