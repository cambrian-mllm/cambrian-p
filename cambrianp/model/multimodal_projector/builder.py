import torch
import torch.nn as nn
import re
import torch.nn.functional as F

from .pooler_projector import PoolerProjector
from .vggt_agg_projector import VGGTAggProjector

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class L2NormLayer(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.dim, self.eps = dim, eps
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

def _maybe_append_norm(modules: list, norm_tag: str | None, hidden_size: int):
    if not norm_tag:
        return
    tag = norm_tag.lower()
    if tag in ("ln", "layernorm"):
        modules.append(nn.LayerNorm(hidden_size))
    elif tag in ("rms", "rmsnorm"):
        modules.append(torch.nn.RMSNorm(hidden_size, eps=1e-6))
    elif tag in ("l2",):
        modules.append(L2NormLayer(dim=-1))
    else:
        raise ValueError(f"Unknown projector norm '{norm_tag}'")

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")
    
    return build_projector(projector_type, config.mm_hidden_size, config.hidden_size, delay_load, **kwargs)


def build_projector(projector_type, mm_hidden_size, hidden_size, delay_load=False, **kwargs):
    # linear / pooler unchanged
    if projector_type == "linear":
        return nn.Linear(mm_hidden_size, hidden_size)

    if projector_type == "pooler":
        return PoolerProjector(mm_hidden_size, hidden_size)

    # Support optional post-norm suffix on MLPs:
    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu(?:_(ln|rms|l2))?$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        norm_tag  = mlp_gelu_match.group(2)  # may be None
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        _maybe_append_norm(modules, norm_tag, hidden_size)
        return nn.Sequential(*modules)

    #   mlp2x_res2x_gelu[_ln|_rms|_l2]
    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu(?:_(ln|rms|l2))?$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        norm_tag  = mlp_gelu_resnet_match.group(3)
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(hidden_size))
        _maybe_append_norm(modules, norm_tag, hidden_size)
        return nn.Sequential(*modules)

    #   mlp2x_e-gelu[_ln|_rms|_l2]
    mlp_e_gelu_match = re.match(r"^mlp(\d+)x_e-gelu(?:_(ln|rms|l2))?$", projector_type)
    if mlp_e_gelu_match:
        mlp_depth = int(mlp_e_gelu_match.group(1))
        norm_tag  = mlp_e_gelu_match.group(2)
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        modules.append(nn.GELU())
        _maybe_append_norm(modules, norm_tag, hidden_size)
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    vggt_agg_proj_match = re.match(r"^vggt_agg_depth(\d+)x(?:_(ln|rms|l2))?$", projector_type)
    if vggt_agg_proj_match:
        depth    = int(vggt_agg_proj_match.group(1))
        norm_tag = vggt_agg_proj_match.group(2)
        core = VGGTAggProjector(input_dim=mm_hidden_size, dim=hidden_size // 2, depth=depth, **kwargs)
        if norm_tag:
            # wrap core with a post-norm
            return nn.Sequential(core, nn.Identity(), *_maybe_norm_seq(norm_tag, hidden_size))
        return core

    raise ValueError(f"Unknown projector type: {projector_type}")

def _maybe_norm_seq(norm_tag, hidden_size):
    tmp = []
    _maybe_append_norm(tmp, norm_tag, hidden_size)
    return tmp