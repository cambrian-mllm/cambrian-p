# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from math import ceil, floor
import numpy as np

from dataclasses import dataclass
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from vggt.train_utils.general import check_and_fix_inf_nan


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, camera=None, depth=None, point=None, track=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track
        
    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # Camera pose loss - if pose encodings are predicted
        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)   
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]   
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)
        
        # Depth estimation loss - if depth maps are predicted
        if "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)

        # 3D point reconstruction loss - if world points are predicted
        if "world_points" in predictions:
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            point_loss = point_loss_dict["loss_conf_point"] + point_loss_dict["loss_reg_point"] + point_loss_dict["loss_grad_point"]
            point_loss = point_loss * self.point["weight"]
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)

        # Tracking loss - not cleaned yet, dirty code is at the bottom of this file
        if "track" in predictions:
            raise NotImplementedError("Track loss is not cleaned up yet")
        
        loss_dict["objective"] = total_loss

        return loss_dict


def compute_camera_loss(
    pred_dict,
    batch_data,
    loss_type="l1",
    gamma=0.6,
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,
    weight_rot=1.0,
    weight_focal=0.5,
    components="both",  # Options: 'all', 'translation', 'rotation', 'focal'
    use_point_masks=True,
    use_scale_alignment=False,   
    use_trajectory_balance=False,  
    rescale_pred_not_gt=False,
    **kwargs
):
    pred_pose_encodings = pred_dict['pose_enc_list']
    scale_factors_list = []

    if not (use_point_masks and 'point_masks' in batch_data):
        # All frames are valid when not using point masks
        B, S = pred_pose_encodings[0].shape[:2]
        valid_frame_mask = torch.ones(B, S, dtype=torch.bool, device=pred_pose_encodings[0].device)
    else:
        point_masks = batch_data['point_masks']
        valid_frame_mask = point_masks.sum(dim=[-1, -2]) > 100
        
    n_stages = len(pred_pose_encodings)

    gt_extrinsics = batch_data['extrinsics']          # [B,S,3,4]
    gt_intrinsics = batch_data['intrinsics']         # [B,S,3,3] or similar
    image_hw = batch_data['images'].shape[-2:]

    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics.float(), gt_intrinsics.float(), image_hw, pose_encoding_type=pose_encoding_type
    )
    
    # Check if we should apply Sim(3) alignment (non-metric datasets only)
    # is_metric = batch_data.get('is_metric_scale', True)
    B, S = pred_pose_encodings[0].shape[:2]
    device = pred_pose_encodings[0].device
    
    #is_metric_scale: [B] bool tensor, True = metric, False = non-metric after collation
    is_metric_scale = batch_data.get('is_metric_scale', None)
    if is_metric_scale is None:
        # Default to all metric if not provided for compatibility
        is_metric_scale = torch.ones(B, dtype=torch.bool, device=device)
 
    # for logging
    is_metric_ratio = is_metric_scale.float().mean().item()
    
    # mask for samples that need alignment (non-metric only)
    # needs_alignment: [B] bool tensor, True = apply scale alignment for this sample
    needs_alignment = use_scale_alignment & (~is_metric_scale)  
    
    mean_traj_length = 1.0
    if use_trajectory_balance or needs_alignment.any():
        from vggt.train_utils.sim3_alignment import compute_trajectory_length
        with torch.no_grad():
            gt_traj_length = compute_trajectory_length(gt_extrinsics)
            mean_traj_length = gt_traj_length.clamp(min=0.1).mean().item()
    
    # Determine which components to compute
    use_translation = components in ['all', 'translation']
    use_rotation = components in ['all', 'rotation']
    use_focal = components in ['all', 'focal']

    total_loss_T = total_loss_R = total_loss_FL = 0.0
    any_valid = valid_frame_mask.any()

    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]
        
        # Apply scale alignment for non-metric datasets ONLY
        if needs_alignment.any():
            # Compute scale s* = argmin ||s*·pred_t - gt_t|| with stop-gradient
            with torch.no_grad():
                scales = compute_scale_only_alignment(
                    pred_pose_stage, gt_pose_encoding, valid_frame_mask, needs_alignment
                )
            scale_factors_list.append(scales[needs_alignment].mean().item())
            
            if rescale_pred_not_gt:
                # rescale pred to GT's scale (gradients flow through pred)
                adjusted_pred = pred_pose_stage.clone()
                adjusted_pred[..., :3] = pred_pose_stage[..., :3] * scales[:, None, None].detach().clamp(min=0.01)
                pred_pose_for_loss = adjusted_pred
                gt_for_loss = gt_pose_encoding
            else:
                # rescale GT to pred's scale (backward compatibility)
                adjusted_gt = gt_pose_encoding.clone()
                adjusted_gt[..., :3] = gt_pose_encoding[..., :3] / scales[:, None, None].detach().clamp(min=0.01)
                pred_pose_for_loss = pred_pose_stage
                gt_for_loss = adjusted_gt
        else:
            pred_pose_for_loss = pred_pose_stage
            gt_for_loss = gt_pose_encoding

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_for_loss * 0).mean()
            loss_R_stage = (pred_pose_for_loss * 0).mean()
            loss_FL_stage = (pred_pose_for_loss * 0).mean()
        else:
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_for_loss[valid_frame_mask].clone(),
                gt_for_loss[valid_frame_mask].clone(),
                loss_type=loss_type
            )

        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average losses
    avg_loss_T = total_loss_T / n_stages if use_translation else 0.0
    avg_loss_R = total_loss_R / n_stages if use_rotation else 0.0
    avg_loss_FL = total_loss_FL / n_stages if use_focal else 0.0


    # Normalize translation loss by trajectory length when use_trajectory_balance=True
    # to balance gradients between large-scale (outdoor/driving) and small-scale (indoor) scenes.
    if use_trajectory_balance and use_translation:
        avg_loss_T = avg_loss_T / mean_traj_length

    # Combine with weights (only for used components)
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )
    
   
    # Compute accuracy metrics
    final_pred_pose = pred_pose_encodings[-1]
    seq_mask = valid_frame_mask.any(dim=1)
    if any_valid:
        selected_pred = final_pred_pose[seq_mask]
        selected_gt = gt_extrinsics[seq_mask]
        error_metrics = compute_camera_accuracy_metrics(
            selected_pred, selected_gt, image_hw, pose_encoding_type, device=final_pred_pose.device
        )
    else:
        device = final_pred_pose.device
        z = torch.tensor(-1.0, device=device)
        error_metrics = {
            "rel_rangle_mean": z, "rel_tangle_mean": z,
            "rel_rangle_median": z, "rel_tangle_median": z,
            "rel_rangle_mean_to1": z, "rel_tangle_mean_to1": z,
            "rel_rangle_median_to1": z, "rel_tangle_median_to1": z,
            "Racc_15": z, "Tacc_15": z, "Racc_15_to1": z, "Tacc_15_to1": z,
        }
        
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL,
        "gt_traj_length": mean_traj_length,
        "is_metric": is_metric_ratio,
        "used_scale_alignment": float(needs_alignment.any()),
        "used_trajectory_balance": float(use_trajectory_balance),
        'align_rescale': np.mean(scale_factors_list) if scale_factors_list else 0.0,
        'traj_length_raw': gt_traj_length.mean().item() if use_trajectory_balance else 0.0,
        "camera_rot_err": error_metrics["rel_rangle_mean"],
        "camera_trans_err": error_metrics["rel_tangle_mean"],
        "camera_rot_err_med": error_metrics["rel_rangle_median"],
        "camera_trans_err_med": error_metrics["rel_tangle_median"],
        "camera_rot_err_to1": error_metrics["rel_rangle_mean_to1"],
        "camera_trans_err_to1": error_metrics["rel_tangle_mean_to1"],
        "Racc_15": error_metrics["Racc_15"],
        "Tacc_15": error_metrics["Tacc_15"],
        "Racc_15_to1": error_metrics["Racc_15_to1"],
        "Tacc_15_to1": error_metrics["Tacc_15_to1"],
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1",batch=None, valid_frame_mask=None):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points']
    gt_points_mask = batch['point_masks']
    
    # Convert to fp32 for stable loss computation
    pred_points = pred_points.float()
    pred_points_conf = pred_points_conf.float()
    
    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")
    gt_points = gt_points.float()
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, use_point_masks=True, **kwargs):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depths']
    
    # Convert to fp32 for stable loss computation
    pred_depth = pred_depth.float()
    pred_depth_conf = pred_depth_conf.float()
    
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth.float()
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    if use_point_masks and 'point_masks' in batch:
        gt_depth_mask = batch['point_masks'].clone()   # [B,S,H,W]
    else:
        # Use all points (mask where gt is finite/valid)
        gt_depth_mask = torch.isfinite(gt_depth).all(dim=-1).clone()

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict

def compute_camera_accuracy_metrics(pred_pose_enc, gt_extrinsics, image_hw, pose_encoding_type="absT_quaR_FoV", device="cuda"):
    """
    Compute rotation and translation error metrics and accuracies.
    
    Args:
        pred_pose_enc: (B, S, D) or (B*S, D) predicted pose encoding 
        gt_extrinsics: (B, S, 3, 4) or (B*S, 3, 4) ground truth extrinsic matrices
        image_hw: Image height and width
        pose_encoding_type: Type of pose encoding
        
    Returns:
        Dict with error metrics (in degrees) and accuracies
    """
    with torch.no_grad():   
        if pred_pose_enc.dim() == 3:  # [B, S, D]
            B, S, D = pred_pose_enc.shape
            
            pred_extrinsics, _ = pose_encoding_to_extri_intri(
                pred_pose_enc, image_hw, pose_encoding_type=pose_encoding_type
            )
            
        else:  # [B*S, D]  
            pred_extrinsics, _ = pose_encoding_to_extri_intri(
                pred_pose_enc, image_hw, pose_encoding_type=pose_encoding_type
            )
            B = 1  # Treat as single batch
            S = pred_extrinsics.shape[0]
            pred_extrinsics = pred_extrinsics.reshape(B, S, 3, 4)
            gt_extrinsics = gt_extrinsics.reshape(B, S, 3, 4)
        
        # add [0, 0, 0, 1] row
        B, S = pred_extrinsics.shape[:2]
        pred_extrinsics_4x4 = torch.zeros(B, S, 4, 4, device=pred_extrinsics.device, dtype=pred_extrinsics.dtype)
        pred_extrinsics_4x4[:, :, :3, :] = pred_extrinsics
        pred_extrinsics_4x4[:, :, 3, 3] = 1.0
        
        gt_extrinsics_4x4 = torch.zeros(B, S, 4, 4, device=gt_extrinsics.device, dtype=gt_extrinsics.dtype)
        gt_extrinsics_4x4[:, :, :3, :] = gt_extrinsics
        gt_extrinsics_4x4[:, :, 3, 3] = 1.0
        
        # Ensure both tensors are in float32 for stable computation
        pred_extrinsics_4x4 = pred_extrinsics_4x4.float()
        gt_extrinsics_4x4 = gt_extrinsics_4x4.float()
        
        rel_rangle_deg, rel_tangle_deg, rel_rangle_deg_to1, rel_tangle_deg_to1 = camera_to_rel_deg(
            pred_extrinsics_4x4, gt_extrinsics_4x4, device
        )
        
        if rel_rangle_deg.numel() > 0:
            rel_rangle_mean = rel_rangle_deg.mean()
            rel_tangle_mean = rel_tangle_deg.mean()
            rel_rangle_median = rel_rangle_deg.median()
            rel_tangle_median = rel_tangle_deg.median()
            
            # Errors relative to first frame
            rel_rangle_mean_to1 = rel_rangle_deg_to1.mean()
            rel_tangle_mean_to1 = rel_tangle_deg_to1.mean()
            rel_rangle_median_to1 = rel_rangle_deg_to1.median()
            rel_tangle_median_to1 = rel_tangle_deg_to1.median()
            
            Racc_15 = (rel_rangle_deg < 15.0).float().mean()
            Tacc_15 = (rel_tangle_deg < 15.0).float().mean()
            Racc_15_to1 = (rel_rangle_deg_to1 < 15.0).float().mean()
            Tacc_15_to1 = (rel_tangle_deg_to1 < 15.0).float().mean()
            
        else:
            # No valid pairs
            rel_rangle_mean = rel_tangle_mean = torch.tensor(0.0, device=device)
            rel_rangle_median = rel_tangle_median = torch.tensor(0.0, device=device)
            rel_rangle_mean_to1 = rel_tangle_mean_to1 = torch.tensor(0.0, device=device)
            rel_rangle_median_to1 = rel_tangle_median_to1 = torch.tensor(0.0, device=device)
            
            Racc_15 = Tacc_15 = torch.tensor(0.0, device=device)
            Racc_15_to1 = Tacc_15_to1 = torch.tensor(0.0, device=device)
    
    return {
        "rel_rangle_mean": rel_rangle_mean,      
        "rel_tangle_mean": rel_tangle_mean,     
        "rel_rangle_median": rel_rangle_median,   
        "rel_tangle_median": rel_tangle_median,  
        "rel_rangle_mean_to1": rel_rangle_mean_to1,
        "rel_tangle_mean_to1": rel_tangle_mean_to1,
        "rel_rangle_median_to1": rel_rangle_median_to1,
        "rel_tangle_median_to1": rel_tangle_median_to1,
        "Racc_15": Racc_15,
        "Tacc_15": Tacc_15,
        "Racc_15_to1": Racc_15_to1,
        "Tacc_15_to1": Tacc_15_to1,
    }
def camera_to_rel_deg(pred_extrinsic, gt_extrinsic, device):
    
    if pred_extrinsic.numel() == 0 or gt_extrinsic.numel() == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([], device=device)
    
    # Ensure float32 for stable computation
    pred_extrinsic = pred_extrinsic.float()
    gt_extrinsic = gt_extrinsic.float()
    
    # Ensure we have the same shape
    if pred_extrinsic.shape != gt_extrinsic.shape:
        min_frames = min(pred_extrinsic.shape[1], gt_extrinsic.shape[1])
        pred_extrinsic = pred_extrinsic[:, :min_frames]
        gt_extrinsic = gt_extrinsic[:, :min_frames]
    
    B, S = pred_extrinsic.shape[:2]
    
    # Extract rotation matrices and translation vectors
    pred_R = pred_extrinsic[..., :3, :3]  # [B, S, 3, 3]
    pred_t = pred_extrinsic[..., :3, 3]   # [B, S, 3]
    gt_R = gt_extrinsic[..., :3, :3]      # [B, S, 3, 3]
    gt_t = gt_extrinsic[..., :3, 3]       # [B, S, 3]
    
    rel_rangle_deg_list = []
    rel_tangle_deg_list = []
    rel_rangle_deg_to1_list = []
    rel_tangle_deg_to1_list = []
    
    # Calculate relative angles for consecutive frames
    for b in range(B):
        for s in range(S - 1):
            # === Consecutive frames (s to s+1) ===
            pred_R1, pred_R2 = pred_R[b, s], pred_R[b, s + 1]
            pred_t1, pred_t2 = pred_t[b, s], pred_t[b, s + 1]
            gt_R1, gt_R2 = gt_R[b, s], gt_R[b, s + 1]
            gt_t1, gt_t2 = gt_t[b, s], gt_t[b, s + 1]
            
            # Relative rotation for w2c: R_rel = R2 @ R1^T
            pred_R_rel = torch.matmul(pred_R2, pred_R1.transpose(-2, -1))
            gt_R_rel = torch.matmul(gt_R2, gt_R1.transpose(-2, -1))
            R_error = torch.matmul(pred_R_rel, gt_R_rel.transpose(-2, -1))
            trace = torch.clamp(R_error.trace(), -1.0, 3.0)
            angle_rad = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
            angle_deg = angle_rad * 180.0 / torch.pi
            rel_rangle_deg_list.append(angle_deg)
            
            # Extract camera centers: C = -R^T @ t
            C1_pred = -torch.matmul(pred_R1.transpose(-1, -2), pred_t1)
            C2_pred = -torch.matmul(pred_R2.transpose(-1, -2), pred_t2)
            C1_gt = -torch.matmul(gt_R1.transpose(-1, -2), gt_t1)
            C2_gt = -torch.matmul(gt_R2.transpose(-1, -2), gt_t2)
            
            # Relative translation in camera 1's frame
            pred_t_rel_local = torch.matmul(pred_R1, C2_pred - C1_pred)
            gt_t_rel_local = torch.matmul(gt_R1, C2_gt - C1_gt)
            
            # Normalize and compute angle
            pred_t_norm = F.normalize(pred_t_rel_local, p=2, dim=-1, eps=1e-8)
            gt_t_norm = F.normalize(gt_t_rel_local, p=2, dim=-1, eps=1e-8)
            cos_sim = torch.clamp(torch.sum(pred_t_norm * gt_t_norm, dim=-1), -1.0, 1.0)
            t_angle_rad = torch.acos(cos_sim)
            t_angle_deg = t_angle_rad * 180.0 / torch.pi
            rel_tangle_deg_list.append(t_angle_deg)
            
            # === Relative to first frame (0 to s+1) ===
            if s + 1 > 0:
                pred_R1_to1, pred_R2_to1 = pred_R[b, 0], pred_R[b, s + 1]
                pred_t1_to1, pred_t2_to1 = pred_t[b, 0], pred_t[b, s + 1]
                gt_R1_to1, gt_R2_to1 = gt_R[b, 0], gt_R[b, s + 1]
                gt_t1_to1, gt_t2_to1 = gt_t[b, 0], gt_t[b, s + 1]
                
                # Relative rotation to first frame
                pred_R_rel_to1 = torch.matmul(pred_R2_to1, pred_R1_to1.transpose(-2, -1))
                gt_R_rel_to1 = torch.matmul(gt_R2_to1, gt_R1_to1.transpose(-2, -1))
                R_error_to1 = torch.matmul(pred_R_rel_to1, gt_R_rel_to1.transpose(-2, -1))
                trace_to1 = torch.clamp(R_error_to1.trace(), -1.0, 3.0)
                angle_rad_to1 = torch.acos(torch.clamp((trace_to1 - 1) / 2, -1.0, 1.0))
                angle_deg_to1 = angle_rad_to1 * 180.0 / torch.pi
                rel_rangle_deg_to1_list.append(angle_deg_to1)
            
                C1_pred_to1 = -torch.matmul(pred_R1_to1.transpose(-1, -2), pred_t1_to1)
                C2_pred_to1 = -torch.matmul(pred_R2_to1.transpose(-1, -2), pred_t2_to1)
                C1_gt_to1 = -torch.matmul(gt_R1_to1.transpose(-1, -2), gt_t1_to1)
                C2_gt_to1 = -torch.matmul(gt_R2_to1.transpose(-1, -2), gt_t2_to1)
                
                pred_t_rel_to1_local = torch.matmul(pred_R1_to1, C2_pred_to1 - C1_pred_to1)
                gt_t_rel_to1_local = torch.matmul(gt_R1_to1, C2_gt_to1 - C1_gt_to1)
                
                pred_t_norm_to1 = F.normalize(pred_t_rel_to1_local, p=2, dim=-1, eps=1e-8)
                gt_t_norm_to1 = F.normalize(gt_t_rel_to1_local, p=2, dim=-1, eps=1e-8)
                cos_sim_to1 = torch.clamp(torch.sum(pred_t_norm_to1 * gt_t_norm_to1, dim=-1), -1.0, 1.0)
                t_angle_rad_to1 = torch.acos(cos_sim_to1)
                t_angle_deg_to1 = t_angle_rad_to1 * 180.0 / torch.pi
                rel_tangle_deg_to1_list.append(t_angle_deg_to1)
    
    if rel_rangle_deg_list:
        rel_rangle_deg = torch.stack(rel_rangle_deg_list)
        rel_tangle_deg = torch.stack(rel_tangle_deg_list)
        rel_rangle_deg_to1 = torch.stack(rel_rangle_deg_to1_list)
        rel_tangle_deg_to1 = torch.stack(rel_tangle_deg_to1_list)
    else:
        rel_rangle_deg = torch.tensor([], device=device)
        rel_tangle_deg = torch.tensor([], device=device)
        rel_rangle_deg_to1 = torch.tensor([], device=device)
        rel_tangle_deg_to1 = torch.tensor([], device=device)
    
    return rel_rangle_deg, rel_tangle_deg, rel_rangle_deg_to1, rel_tangle_deg_to1

def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss

def compute_scale_only_alignment(pred_pose_enc, gt_pose_enc, valid_mask, needs_alignment):
 
    B = pred_pose_enc.shape[0]
    device = pred_pose_enc.device
    
    pred_t = pred_pose_enc[..., :3]
    gt_t = gt_pose_enc[..., :3]
    
    mask = valid_mask.unsqueeze(-1).float()
    
    numerator = (pred_t * gt_t * mask).sum(dim=[1, 2])
    denominator = (pred_t * pred_t * mask).sum(dim=[1, 2])
    
    scales = (numerator / (denominator + 1e-8)).clamp(min=0.01, max=100.0)
    
    valid_counts = valid_mask.sum(dim=1)
    should_scale = needs_alignment & (valid_counts >= 2)
    
    return torch.where(should_scale, scales, torch.ones(B, device=device))

def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


########################################################################################
########################################################################################

# Dirty code for tracking loss:

########################################################################################
########################################################################################

'''
def _compute_losses(self, coord_preds, vis_scores, conf_scores, batch):
    """Compute tracking losses using sequence_loss"""
    gt_tracks = batch["tracks"]  # B, S, N, 2
    gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N

    # if self.training and hasattr(self, "train_query_points"):
    train_query_points = coord_preds[-1].shape[2]
    gt_tracks = gt_tracks[:, :, :train_query_points]
    gt_tracks = check_and_fix_inf_nan(gt_tracks, "gt_tracks", hard_max=None)

    gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]

    # Create validity mask that filters out tracks not visible in first frame
    valids = torch.ones_like(gt_track_vis_mask)
    mask = gt_track_vis_mask[:, 0, :] == True
    valids = valids * mask.unsqueeze(1)



    if not valids.any():
        print("No valid tracks found in first frame")
        print("seq_name: ", batch["seq_name"])
        print("ids: ", batch["ids"])
        print("time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        dummy_coord = coord_preds[0].mean() * 0          # keeps graph & grads
        dummy_vis = vis_scores.mean() * 0
        if conf_scores is not None:
            dummy_conf = conf_scores.mean() * 0
        else:
            dummy_conf = 0
        return dummy_coord, dummy_vis, dummy_conf                # three scalar zeros


    # Compute tracking loss using sequence_loss
    track_loss = sequence_loss(
        flow_preds=coord_preds,
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
        **self.loss_kwargs
    )

    vis_loss = F.binary_cross_entropy_with_logits(vis_scores[valids], gt_track_vis_mask[valids].float())

    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=None)


    # within 3 pixels
    if conf_scores is not None:
        gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(conf_scores[valids], gt_conf_mask[valids].float())
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=None)
    else:
        conf_loss = 0

    return track_loss, vis_loss, conf_loss



def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0,
                       mean,
                       torch.zeros_like(mean))
    return mean


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, huber=False, delta=10, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)

        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid_points > 2:
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=None)
                flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss
'''