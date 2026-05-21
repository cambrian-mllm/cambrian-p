"""
Scale alignment and trajectory utilities for non-metric datasets.

Provides Sim(3) alignment (scale + rotation + translation) for aligning
predicted camera poses to ground truth when metric scale is not available.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_trajectory_length(extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Compute the total trajectory length from camera extrinsics.
    
    Args:
        extrinsics: Camera extrinsics [B, S, 3, 4] or [B, S, 4, 4] (world-to-camera)
        
    Returns:
        trajectory_length: [B] total distance traveled by camera center
    """
    B, S = extrinsics.shape[:2]
    
    # Extract rotation and translation
    R = extrinsics[..., :3, :3]  # [B, S, 3, 3]
    t = extrinsics[..., :3, 3]   # [B, S, 3]
    
    # Camera center in world coordinates: C = -R^T @ t
    # For world-to-camera: R @ X_world + t = X_cam
    # So X_world = R^T @ (X_cam - t) when X_cam = 0: C = -R^T @ t
    camera_centers = -torch.einsum('bsij,bsj->bsi', R.transpose(-1, -2), t)  # [B, S, 3]
    
    # Compute distances between consecutive camera centers
    if S < 2:
        return torch.zeros(B, device=extrinsics.device, dtype=extrinsics.dtype)
    
    diffs = camera_centers[:, 1:] - camera_centers[:, :-1]  # [B, S-1, 3]
    distances = torch.norm(diffs, dim=-1)  # [B, S-1]
    trajectory_length = distances.sum(dim=-1)  # [B]
    
    return trajectory_length


def extract_camera_centers(extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Extract camera centers from world-to-camera extrinsics.
    
    Args:
        extrinsics: [B, S, 3, 4] or [B, S, 4, 4] world-to-camera matrices
        
    Returns:
        camera_centers: [B, S, 3] camera positions in world coordinates
    """
    R = extrinsics[..., :3, :3]  # [B, S, 3, 3]
    t = extrinsics[..., :3, 3]   # [B, S, 3]
    
    # C = -R^T @ t
    camera_centers = -torch.einsum('...ij,...j->...i', R.transpose(-1, -2), t)
    return camera_centers


def procrustes_analysis(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    allow_scale: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Procrustes analysis to find optimal rotation, translation, and scale.
    
    Finds R, t, s such that: s * R @ source + t ≈ target
    
    Args:
        source: [N, 3] source points
        target: [N, 3] target points
        weights: [N] optional weights for each point
        allow_scale: whether to compute scale (True) or fix to 1.0
        
    Returns:
        R: [3, 3] rotation matrix
        t: [3] translation vector
        s: scalar scale factor
    """
    assert source.shape == target.shape
    orig_dtype = source.dtype
    source = source.float()
    target = target.float()
    if weights is not None:
        weights = weights.float()
    N = source.shape[0]
    
    if weights is None:
        weights = torch.ones(N, device=source.device, dtype=source.dtype)
    
    weights = weights / weights.sum()  # Normalize weights
    
    # Compute weighted centroids
    source_centroid = (weights.unsqueeze(-1) * source).sum(dim=0)
    target_centroid = (weights.unsqueeze(-1) * target).sum(dim=0)
    
    # Center the points
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    
    # Compute weighted covariance matrix
    W = torch.diag(weights)
    H = source_centered.T @ W @ target_centered  # [3, 3]
    
    # SVD for rotation
    U, S_vals, Vt = torch.linalg.svd(H)
    
    # Ensure proper rotation (det = 1)
    det = torch.det(Vt.T @ U.T)
    D = torch.eye(3, device=source.device, dtype=source.dtype)
    D[2, 2] = det.sign()
    
    R = Vt.T @ D @ U.T
    
    # Compute scale
    if allow_scale:
        source_var = (weights.unsqueeze(-1) * (source_centered ** 2)).sum()
        target_var = (weights.unsqueeze(-1) * (target_centered ** 2)).sum()
        
        # Trace of R @ H gives the correlation
        rotated_source = source_centered @ R.T
        correlation = (weights.unsqueeze(-1) * rotated_source * target_centered).sum()
        
        if source_var > 1e-8:
            s = correlation / source_var
            s = s.clamp(min=0.01, max=100.0)  # Reasonable scale bounds
        else:
            s = torch.ones(1, device=source.device, dtype=source.dtype)
    else:
        s = torch.ones(1, device=source.device, dtype=source.dtype)
    
    # Compute translation
    t = target_centroid - s * (R @ source_centroid)
    
    return R, t, s


def sim3_align_trajectories(
    pred_extrinsics: torch.Tensor,
    gt_extrinsics: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align predicted camera trajectory to ground truth using Sim(3).
    
    Finds scale s, rotation R, translation t such that for camera centers:
        C_aligned = s * R_align @ C_pred + t_align ≈ C_gt
    
    The transformation of extrinsics follows from:
        X_aligned_world = s * R_align @ X_pred_world + t_align
    
    For world-to-camera extrinsic [R_pred | t_pred]:
        X_cam = R_pred @ X_pred_world + t_pred
    
    After Sim(3) alignment of world coordinates:
        X_cam = R_new @ X_aligned_world + t_new
        
    Solving gives:
        R_new = R_pred @ R_align^T
        t_new = s * t_pred - R_new @ t_align
    
    Args:
        pred_extrinsics: [B, S, 3, 4] predicted world-to-camera extrinsics
        gt_extrinsics: [B, S, 3, 4] ground truth world-to-camera extrinsics
        valid_mask: [B, S] optional mask for valid frames
        
    Returns:
        aligned_pred_extrinsics: [B, S, 3, 4] aligned predictions
        scale_factors: [B] scale factors used for alignment
    """
    B, S = pred_extrinsics.shape[:2]
    device = pred_extrinsics.device
    dtype = pred_extrinsics.dtype
    
    pred_extrinsics = pred_extrinsics.float()
    gt_extrinsics = gt_extrinsics.float()
    
    # Extract camera centers
    pred_centers = extract_camera_centers(pred_extrinsics)  # [B, S, 3]
    gt_centers = extract_camera_centers(gt_extrinsics)      # [B, S, 3]
    
    aligned_extrinsics = pred_extrinsics.clone()
    scale_factors = torch.ones(B, device=device, dtype=torch.float32)
    
    for b in range(B):
        if valid_mask is not None:
            mask_b = valid_mask[b]
            if mask_b.sum() < 3:
                continue
            pred_pts = pred_centers[b, mask_b]
            gt_pts = gt_centers[b, mask_b]
        else:
            pred_pts = pred_centers[b]
            gt_pts = gt_centers[b]
        
        if pred_pts.shape[0] < 3:
            continue
        
        # Procrustes analysis to find Sim(3) alignment
        # Finds R_align, t_align, s such that: s * R_align @ pred_pts + t_align ≈ gt_pts
        R_align, t_align, s_align = procrustes_analysis(
            pred_pts, gt_pts, allow_scale=True
        )
        
        scale_factors[b] = s_align
        
        # Apply alignment to extrinsics
        R_pred = pred_extrinsics[b, :, :3, :3]  # [S, 3, 3]
        t_pred = pred_extrinsics[b, :, :3, 3]   # [S, 3]
        
        R_align_inv = R_align.T  # [3, 3]
        
        R_new = torch.einsum('sij,jk->sik', R_pred, R_align_inv)
        
        t_new = s_align * t_pred - torch.einsum('sij,j->si', R_new, t_align)
        
        aligned_extrinsics[b, :, :3, :3] = R_new
        aligned_extrinsics[b, :, :3, 3] = t_new
    
    return aligned_extrinsics.to(dtype), scale_factors.to(dtype)
