import os
import json
import numpy as np
import torch
from PIL import Image
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from typing import Optional, List, Dict, Any, Tuple, Union
import re
import wandb
import matplotlib.pyplot as plt


_PAIR_RE = r'\bframe\s+(\d+)\s*to\s*(?:frame\s+)?(\d+)\s*:\s*([^|]+)'
_FALLBACK_RE = r'F\s*([0-9]+)\s*:\s*(.+?)(?=F\s*[0-9]+:|$)'


class CameraExtrinsicToText:
    """Convert camera extrinsics (world→camera) to text.

    Reference policy (fixed):
      - reference_frame == 'first':     T_ref(i) = E0 @ inv(Ei)         (express i in Frame-0 camera coords)
      - reference_frame == 'previous':  T_ref(i) = E_{i-1} @ inv(Ei)    (express i in previous camera coords)

    Text strategies:
      - 'metric' (default): numeric distances; rotations quantized to {±15, ±30, ±45}°.
                            Uses plain words: right/up, move forward/backward, pan/tilt (roll optional).
      - 'simple'          : no numbers; easy words. Moves: "move right/up/forward/backward".
                            Rotations: "turn left/right" (yaw), "look up/down" (pitch), "lean left/right" (roll, optional).
      - 'professional'    : filmmaking terms. Moves: truck (left/right), pedestal (up/down), dolly (in/out).
                            Rotations: pan (yaw), tilt (pitch, degrees kept), Dutch-angle term for roll (optional; Table A1 bins).

    Roll usage:
      - `use_roll` controls whether a roll phrase is emitted per style.
        * bool -> apply to all styles
        * dict -> keys in {'metric','simple','professional'}; unspecified default to False
        * default: omit roll for all styles (easier supervision)
    """

    def __init__(
        self,
        mode: str = 'description',
        reference_frame: str = 'first',
        precision: int = 1,
        text_strategy: str = 'metric',
        use_roll: Union[bool, Dict[str, bool], None] = None,
        yaw_sign: int = -1,
        pitch_sign: int = -1,
    ):
        assert mode in {'description', 'quaternion', 'matrix'}
        assert reference_frame in {'first', 'previous'}
        assert text_strategy in {'metric', 'simple', 'professional'}
        self.mode = mode
        self.reference_frame = reference_frame
        self.precision = precision
        self.text_strategy = text_strategy

        # Configure per-style roll emission
        if use_roll is None:
            self.use_roll: Dict[str, bool] = {'metric': False, 'simple': False, 'professional': False}
        elif isinstance(use_roll, bool):
            self.use_roll = {'metric': use_roll, 'simple': use_roll, 'professional': use_roll}
        else:
            self.use_roll = {
                'metric': use_roll.get('metric', False),
                'simple': use_roll.get('simple', False),
                'professional': use_roll.get('professional', False),
            }
        assert yaw_sign in (-1, +1)
        self.yaw_sign = yaw_sign
        assert pitch_sign in (-1, +1)
        self.pitch_sign = pitch_sign

        # Table A1 bins (for professional roll wording)
        self.roll_terms = [
            (-45, -20, "large counterclockwise Dutch angle"),
            (-20,  -5, "small counterclockwise Dutch angle"),
            ( -5,   5, "near level shot"),
            (  5,  20, "small clockwise Dutch angle"),
            ( 20,  45, "large clockwise Dutch angle"),
        ]
        self.pitch_terms = [
            (-45, -20, "large tilt-down"),
            (-20,  -5, "small tilt-down"),
            ( -5,   5, "near straight-on shot"),
            (  5,  20, "small tilt-up"),
            ( 20,  45, "large tilt-up"),
        ]

    def _yaw_label(self, yaw_deg: float) -> str:
        y = self.yaw_sign * yaw_deg
        return 'left' if y > 0 else 'right'

    def _pitch_label(self, pitch_deg: float) -> str:
        p = self.pitch_sign * pitch_deg
        return 'down' if p > 0 else 'up'

    def process_extrinsics(self, extrinsics: List[np.ndarray]) -> str:
        if len(extrinsics) == 0:
            return ""

        # Ensure 4x4 world→camera
        E = []
        for e in extrinsics:
            e_np = self._to_numpy(e)
            if e_np.shape == (3, 4):
                M = np.eye(4, dtype=float)
                M[:3, :] = e_np
                e_np = M
            E.append(e_np.astype(float))

        if self.mode == 'description':
            return (
                self._describe_relative_to_first(E)
                if self.reference_frame == 'first'
                else self._describe_relative_to_previous(E)
            )
        elif self.mode == 'quaternion':
            return self._format_quaternion(E)
        else:
            return self._format_matrix(E)

    def _format_quaternion(self, E):
        out = []
        out.append("frame 0: Q:[0.000,0.000,0.000,1.000] T:[0.000,0.000,0.000]")
        for i, Ei in enumerate(E):
            if i == 0:
                continue
            T = (E[0] @ np.linalg.inv(Ei)) if self.reference_frame == 'first' else (E[i-1] @ np.linalg.inv(Ei))
            q = R.from_matrix(T[:3, :3]).as_quat()
            t = T[:3, 3]
            out.append(
                "frame {}: Q:[{:.3f},{:.3f},{:.3f},{:.3f}] T:[{:.3f},{:.3f},{:.3f}]"
                .format(i, q[0], q[1], q[2], q[3], t[0], t[1], t[2])
            )
        return " ".join(out)

    def _format_matrix(self, E):
        out = []
        out.append("frame 0: R:[1.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1.000] T:[0.000,0.000,0.000]")
        for i, Ei in enumerate(E):
            if i == 0:
                continue
            T = (E[0] @ np.linalg.inv(Ei)) if self.reference_frame == 'first' else (E[i-1] @ np.linalg.inv(Ei))
            r = T[:3, :3].reshape(-1)
            t = T[:3, 3]
            out.append(
                "frame {}: R:[{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}] T:[{:.3f},{:.3f},{:.3f}]"
                .format(i, *r, t[0], t[1], t[2])
            )
        return " ".join(out)
    
    def _describe_relative_to_first(self, E):
        lines = []
        for i in range(1, len(E)):
            T = E[0] @ np.linalg.inv(E[i])  # cam0 -> cami
            trans = T[:3, 3]
            yaw, pitch, roll = self._euler_yxz_deg(T[:3, :3], quantize=True)
            parts = self._nl_parts(trans, yaw, pitch, roll, first_mode=True)
            lines.append(f"frame 0 to {i}: {parts if parts else 'Hold'}")
        return " | ".join(lines)

    def _describe_relative_to_previous(self, E):
        lines = []
        for i in range(1, len(E)):
            T = E[i - 1] @ np.linalg.inv(E[i])  # cam(i-1) -> cami
            trans = T[:3, 3]
            yaw, pitch, roll = self._euler_yxz_deg(T[:3, :3], quantize=True)
            parts = self._nl_parts(trans, yaw, pitch, roll, first_mode=False)
            lines.append(f"frame {i-1} to {i}: {parts if parts else 'Hold'}")
        return " | ".join(lines)

    def _nl_parts(self, t, yaw, pitch, roll, *, first_mode: bool):
        if self.text_strategy == 'simple':
            return self._simple_text(t, yaw, pitch, roll, first_mode=first_mode)
        elif self.text_strategy == 'professional':
            return self._professional_text(t, yaw, pitch, roll, first_mode=first_mode)
        else:
            return self._metric_text(t, yaw, pitch, roll, first_mode=first_mode)

    def _metric_text(self, t, yaw, pitch, roll, *, first_mode: bool):
        th = 0.1
        parts = []
        if abs(t[0]) > th:
            parts.append(f"{'right' if t[0] > 0 else 'left'} {abs(t[0]):.{self.precision}f}m")
        if abs(t[1]) > th:
            parts.append(f"{'down' if t[1] > 0 else 'up'} {abs(t[1]):.{self.precision}f}m")
        if abs(t[2]) > th:
            parts.append(f"move {'forward' if t[2] > 0 else 'backward'} {abs(t[2]):.{self.precision}f}m")

        if abs(pitch) >= 15:
            parts.append(f"tilt {self._pitch_label(pitch)} {abs(int(pitch))}°")
        if self.use_roll['metric'] and abs(roll) >= 15:
            parts.append(f"roll {abs(int(roll))}°")
        if abs(yaw) >= 15:
            parts.append(f"pan {self._yaw_label(yaw)} {abs(int(yaw))}°")
        return ", ".join(parts)

    def _simple_text(self, t, yaw, pitch, roll, *, first_mode: bool):
        th = 0.1
        moves = []
        if abs(t[0]) > th:
            moves.append("right" if t[0] > 0 else "left")
        if abs(t[1]) > th:
            moves.append("down" if t[1] > 0 else "up")
        if abs(t[2]) > th:
            moves.append("forward" if t[2] > 0 else "backward")

        phrases = []
        if moves:
            phrases.append("move " + ", ".join(moves))

        if abs(yaw) >= 15:
            phrases.append("turn " + self._yaw_label(yaw))
        if abs(pitch) >= 15:
            phrases.append("look " + self._pitch_label(pitch))
        if self.use_roll['simple'] and abs(roll) >= 15:
            phrases.append("lean right" if roll > 0 else "lean left")

        return ", ".join(phrases)

    def _professional_text(self, t, yaw, pitch, roll, *, first_mode: bool):
        th = 0.1
        parts = []
        if abs(t[0]) > th:
            parts.append(f"{'truck right' if t[0] > 0 else 'truck left'} {abs(t[0]):.{self.precision}f}m")
        if abs(t[1]) > th:
            parts.append(f"{'pedestal down' if t[1] > 0 else 'pedestal up'} {abs(t[1]):.{self.precision}f}m")
        if abs(t[2]) > th:
            parts.append(f"{'dolly in' if t[2] > 0 else 'dolly out'} {abs(t[2]):.{self.precision}f}m")

        if abs(yaw) >= 15:
            parts.append(f"pan {self._yaw_label(yaw)} {abs(int(yaw))}°")
        if abs(pitch) >= 15:
            parts.append(f"tilt {self._pitch_label(pitch)} {abs(int(pitch))}°")
        if self.use_roll['professional']:
            term_roll = self._table_term(self.roll_terms, roll)
            if term_roll and ("near level" not in term_roll):
                parts.append(term_roll)

        return ", ".join(parts)

    def _euler_yxz_deg(self, Rm, quantize=True):
        yaw, pitch, roll = R.from_matrix(Rm).as_euler('yxz', degrees=True)
        if quantize:
            yaw = self._q15(yaw); pitch = self._q15(pitch); roll = self._q15(roll)
        return yaw, pitch, roll

    @staticmethod
    def _q15(a: float) -> float:
        """Quantize to nearest 15°, clamp to ±75° and snap tiny to 0."""
        q = 15.0 * np.round(a / 15.0)
        q = float(np.clip(q, -75.0, 75.0))
        return 0.0 if abs(q) < 1e-6 else q

    def _table_term(self, bins, angle):
        a = angle
        for lo, hi, name in bins:
            if lo <= a <= hi:
                return name
        return None

    def _to_numpy(self, x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().float().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)


class CameraTrajectoryDebugger:    
    _sample_counter = 0
    
    def __init__(self, save_dir=None):
        if save_dir is None:
            save_dir = os.environ.get("CAM_TRAJ_VIS_DIR", "./cam_traj_vis_tmp")
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
    
    def save_visualization(
        self,
        rec_views,
        generated_text=None,
        mode='quaternion',
        reference_frame='first',
        text_strategy=None,
        step=None,
        sample_idx=0,
        save_frames=True
    ):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        step_str = f"_step{step}" if step is not None else ""
        folder_name = f"{mode}_{reference_frame}_{timestamp}{step_str}"
        save_dir = os.path.join(self.save_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        metadata = {
            'mode': mode,
            'reference_frame': reference_frame,
            'step': step,
            'timestamp': timestamp,
            'text_strategy': text_strategy,
            'num_frames': len(rec_views.get('extrinsics', [])) if 'extrinsics' in rec_views else 0,
        }
        
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        rec_views_path = os.path.join(save_dir, 'rec_views.npz')
        
        rec_data_to_save = {}
        string_data = {}
        
        for key, value in rec_views.items():
            if key in ['seq_name', 'basenames', 'image_paths', 'sample_mode']:
                string_data[key] = value
            elif isinstance(value, torch.Tensor):
                if value.dtype == torch.bfloat16:
                    rec_data_to_save[key] = value.float().cpu().numpy()
                else:
                    rec_data_to_save[key] = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                rec_data_to_save[key] = value
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                try:
                    rec_data_to_save[key] = np.array(value)
                except:
                    string_data[key] = value
            elif isinstance(value, (int, float, bool)):
                rec_data_to_save[key] = value
        
        np.savez_compressed(rec_views_path, **rec_data_to_save)
        
        with open(os.path.join(save_dir, 'string_metadata.json'), 'w') as f:
            json.dump(string_data, f, indent=2)
        
        if 'extrinsics' in rec_views:
            extrinsics = rec_views['extrinsics']
            if isinstance(extrinsics, torch.Tensor):
                if extrinsics.dtype == torch.bfloat16:
                    extrinsics = extrinsics.float().cpu().numpy()
                else:
                    extrinsics = extrinsics.cpu().numpy()
            np.savez(os.path.join(save_dir, 'extrinsics.npz'), extrinsics=extrinsics)
        
        if generated_text:
            with open(os.path.join(save_dir, 'generated.txt'), 'w') as f:
                f.write(generated_text)
        
        if save_frames and 'images' in rec_views:
            frames_dir = os.path.join(save_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            
            images = rec_views['images']
            
            if isinstance(images, torch.Tensor):
                if images.dtype == torch.bfloat16:
                    images = images.float().cpu().numpy()
                else:
                    images = images.cpu().numpy()
            elif not isinstance(images, np.ndarray):
                images = np.array(images)
            
            for i in range(images.shape[0]):
                if images.shape[1] == 3 and images.ndim >= 3:  
                    frame = images[i].transpose(1, 2, 0)
                else:
                    frame = images[i]
                
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
                
                img = Image.fromarray(frame)
                img.save(os.path.join(frames_dir, f'frame_{i:03d}.jpg'))
        
        return save_dir
    
    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.float().cpu().numpy()
        return np.asarray(tensor)


class RecHeadPoseVisualizer:
    def __init__(self, config):
        self.config = config
        self.pose_encoding_type = getattr(config, "pose_encoding_type", "absT_quaR_FoV")
        self.viz_mode = getattr(config, 'rec_pose_viz_mode', 'quaternion')
        
    def visualize_rec_head_poses(
        self,
        predictions: Dict,
        rec_views: Dict,
        step: int,
    ):
        try:
            from cambrianp.camera_trajectory_utils import CameraTrajectoryVisualizerWandb
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            
            # Get GT extrinsics (already normalized!)
            gt_extrinsics = rec_views.get('extrinsics', None)
            if gt_extrinsics is None:
                return
            
            # Get predictions
            if 'pose_enc_list' in predictions:
                pred_pose_enc = predictions['pose_enc_list'][-1]
            elif 'pose_enc' in predictions:
                pred_pose_enc = predictions['pose_enc']
            else:
                return
            
            # Convert predictions to extrinsics
            # CRITICAL: This should output in the SAME normalized space as training
            image_hw = rec_views['images'].shape[-2:]
            pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
                pred_pose_enc, image_hw, 
                pose_encoding_type=self.pose_encoding_type
            )
            
            # ===== COORDINATE FRAME VERIFICATION =====
            # Check if GT and pred are in similar scale ranges
            gt_scale = self._compute_pose_scale(gt_extrinsics)
            pred_scale = self._compute_pose_scale(pred_extrinsics)
            
            scale_ratio = pred_scale / (gt_scale + 1e-8)
            
            if scale_ratio > 10.0 or scale_ratio < 0.1:
                # Try to fix by normalizing pred to match GT scale
                if getattr(self.config, 'auto_fix_scale_mismatch', True):
                    print(f"   Applying scale correction...")
                    pred_extrinsics = self._normalize_to_gt_scale(
                        pred_extrinsics, gt_extrinsics
                    )
            
            # Create visualizer
            visualizer = CameraTrajectoryVisualizerWandb(
                mode=self.viz_mode,
                reference_frame='first',
                text_strategy='simple'
            )
            
            # Visualize first sample in batch
            b = 0
            gt = gt_extrinsics[b] if gt_extrinsics.dim() > 3 else gt_extrinsics
            pred = pred_extrinsics[b] if pred_extrinsics.dim() > 3 else pred_extrinsics
            
            # Convert to [N, 4, 4] format for visualizer
            gt_44 = self._to_homogeneous(gt)
            pred_44 = self._to_homogeneous(pred)
            
            # Log metrics
            pose_metrics = self._log_pose_metrics(gt_44, pred_44, step)
            
            # Visualize - pass metrics to log_to_wandb
            visualizer.log_to_wandb(
                gt_extrinsics=gt_44,
                pred_extrinsics=pred_44,
                depths=rec_views['depths'][b] if rec_views.get('depths') is not None else None,
                intrinsics=rec_views['intrinsics'][b] if rec_views.get('intrinsics') is not None else None,
                images=rec_views['images'][b] if rec_views.get('images') is not None else None,
                world_points=rec_views['world_points'][b] if rec_views.get('world_points') is not None else None,
                point_masks=rec_views['point_masks'][b] if rec_views.get('point_masks') is not None else None,
                prefix="rec_head_poses",
                max_cols=8,
                Racc_15=pose_metrics.get("Racc_15"),
                Tacc_15=pose_metrics.get("Tacc_15"),
                step=step,
            )
                
        except Exception as e:
            print(f"❌ Rec head pose viz error: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_pose_scale(self, extrinsics: torch.Tensor) -> float:
        if extrinsics.dim() == 4:
            # [B, S, 3/4, 4] -> take first batch
            extrinsics = extrinsics[0]
        
        # Extract translations
        if extrinsics.shape[-2] == 3:
            # [S, 3, 4] -> translations are [:, :, 3]
            translations = extrinsics[:, :3, 3]
        else:
            # [S, 4, 4] -> translations are [:, :3, 3]
            translations = extrinsics[:, :3, 3]
        
        # Compute pairwise distances
        if translations.shape[0] > 1:
            dists = torch.norm(translations[1:] - translations[:-1], dim=-1)
            return float(dists.mean().item())
        else:
            return float(torch.norm(translations[0]).item())
    
    def _normalize_to_gt_scale(
        self,
        pred_extrinsics: torch.Tensor,
        gt_extrinsics: torch.Tensor
    ) -> torch.Tensor:
        pred_scale = self._compute_pose_scale(pred_extrinsics)
        gt_scale = self._compute_pose_scale(gt_extrinsics)
        
        scale_factor = gt_scale / (pred_scale + 1e-8)
        
        # Apply scale to translations only
        pred_normalized = pred_extrinsics.clone()
        if pred_normalized.shape[-2] == 3:
            # [B, S, 3, 4] format
            pred_normalized[..., :3, 3] *= scale_factor
        else:
            # [B, S, 4, 4] format
            pred_normalized[..., :3, 3] *= scale_factor
        
        return pred_normalized
    
    def _to_homogeneous(self, extrinsics: torch.Tensor) -> np.ndarray:

        # Convert BFloat16 to Float32 for NumPy compatibility
        if extrinsics.dtype == torch.bfloat16:
            extrinsics = extrinsics.float()
        
        extrinsics_np = extrinsics.detach().cpu().numpy()
        
        if extrinsics_np.shape[-2] == 3:
            # [S, 3, 4] -> [S, 4, 4]
            S = extrinsics_np.shape[0]
            result = np.zeros((S, 4, 4), dtype=np.float32)
            result[:, :3, :4] = extrinsics_np
            result[:, 3, 3] = 1.0
            return result
        else:
            return extrinsics_np.astype(np.float32)
    
    def _log_pose_metrics(
        self,
        gt_poses: np.ndarray,
        pred_poses: np.ndarray,
        step: int,
    ) -> dict:
        try:
            import wandb
            import torch.nn.functional as F
            
            # Translation error (L2 distance)
            gt_trans = gt_poses[:, :3, 3]
            pred_trans = pred_poses[:, :3, 3]
            trans_error = np.linalg.norm(gt_trans - pred_trans, axis=-1).mean()
            
            # Rotation error (geodesic distance)
            rot_errors = []
            for i in range(len(gt_poses)):
                R_gt = gt_poses[i, :3, :3]
                R_pred = pred_poses[i, :3, :3]
                
                trace = np.trace(R_gt.T @ R_pred)
                cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                rot_errors.append(np.degrees(angle))
            
            rot_error = np.mean(rot_errors)
            
            # Convert to torch tensors for accuracy metrics
            gt_tensor = torch.from_numpy(gt_poses).float().unsqueeze(0)
            pred_tensor = torch.from_numpy(pred_poses).float().unsqueeze(0)
            
            Racc_15, Tacc_15 = self._compute_accuracy_metrics(pred_tensor, gt_tensor)

            # Log to wandb (rank 0 only)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                wandb.log({
                    "rec_pose_metrics/translation_error_m": trans_error,
                    "rec_pose_metrics/rotation_error_deg": rot_error,
                    "rec_pose_metrics/Racc_15": Racc_15,
                    "rec_pose_metrics/Tacc_15": Tacc_15,
                    "rec_pose_metrics/gt_scale": self._compute_pose_scale(torch.from_numpy(gt_poses)),
                    "rec_pose_metrics/pred_scale": self._compute_pose_scale(torch.from_numpy(pred_poses)),
                }, step=step)
            
            # Return metrics for visualization
            return {
                "Racc_15": Racc_15,
                "Tacc_15": Tacc_15,
                "trans_error": trans_error,
                "rot_error": rot_error,
            }
                
        except Exception as e:
            print(f"Failed to log pose metrics: {e}")
            return {}
            
    def _compute_accuracy_metrics(
        self,
        pred_extrinsics: torch.Tensor,
        gt_extrinsics: torch.Tensor
    ) -> tuple:
        """
        Compute Racc_15 and Tacc_15 metrics.
        
        Args:
            pred_extrinsics: [B, S, 4, 4] predicted extrinsics
            gt_extrinsics: [B, S, 4, 4] ground truth extrinsics
            
        Returns:
            Racc_15: Fraction of frames with rotation error < 15 degrees
            Tacc_15: Fraction of frames with translation direction error < 15 degrees
        """
        import torch.nn.functional as F
        
        B, S = pred_extrinsics.shape[:2]
        device = pred_extrinsics.device
        
        # Extract rotation matrices and translation vectors
        pred_R = pred_extrinsics[..., :3, :3]  # [B, S, 3, 3]
        pred_t = pred_extrinsics[..., :3, 3]   # [B, S, 3]
        gt_R = gt_extrinsics[..., :3, :3]      # [B, S, 3, 3]
        gt_t = gt_extrinsics[..., :3, 3]       # [B, S, 3]
        
        rel_rangle_deg_list = []
        rel_tangle_deg_list = []
        
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
        
        if rel_rangle_deg_list:
            rel_rangle_deg = torch.stack(rel_rangle_deg_list)
            rel_tangle_deg = torch.stack(rel_tangle_deg_list)
            
            Racc_15 = (rel_rangle_deg < 15.0).float().mean().item()
            Tacc_15 = (rel_tangle_deg < 15.0).float().mean().item()
        else:
            Racc_15 = 0.0
            Tacc_15 = 0.0
        
        return Racc_15, Tacc_15


class CameraTrajectoryVisualizerWandb:
    def __init__(self, mode='quaternion', reference_frame='first', text_strategy='simple'):
        self.mode = mode
        self.reference_frame = reference_frame
        self.text_strategy = text_strategy

    def _to_numpy(self, tensor, dtype=None):
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.bfloat16 or dtype == 'float':
                tensor = tensor.float()
            elif dtype == 'bool':
                tensor = tensor.bool()
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def _fmt(self, x, prec=3):
        if abs(x) < 10 ** (-(prec + 1)):
            x = 0.0
        return f"{x:.{prec}f}"

    def _normalize_extrinsics_44(self, extrinsics: np.ndarray) -> np.ndarray:
        """
        Ensure extrinsics is [N, 4, 4] world->cam.
        Accepts [N, 3, 4] or [N, 4, 4]; also accepts torch tensors and batched tensors.
        """
        E = self._to_numpy(extrinsics, dtype='float')
        if E.ndim == 4:  # [B, N, 3, 4] or [B, N, 4, 4]
            E = E[0]
        if E.shape[-2:] == (3, 4):
            out = np.tile(np.eye(4, dtype=np.float32), (E.shape[0], 1, 1))
            out[:, :3, :4] = E
            return out
        if E.shape[-2:] == (4, 4):
            return E[:, :4, :4].astype(np.float32)
        raise ValueError(f"Unexpected extrinsics shape: {E.shape}")

    def extrinsics_to_text(self, extrinsics, mode=None):
        mode = mode or self.mode
        E44 = self._normalize_extrinsics_44(extrinsics)
        N = E44.shape[0]
        out = []

        if mode in ('matrix', 'quaternion'):
            for i in range(N):
                if i == 0:
                    if mode == 'matrix':
                        out.append("Frame 0: Starting position (R=I, T=[0,0,0])")
                    else:
                        out.append("Frame 0: Starting position (Q=[0,0,0,1], T=[0,0,0])")
                    continue

                if self.reference_frame == 'first':
                    T = E44[0] @ np.linalg.inv(E44[i])
                    prefix = f"Frame 0 to Frame {i}: "
                else:
                    T = E44[i - 1] @ np.linalg.inv(E44[i])
                    prefix = f"Frame {i-1} to Frame {i}: "

                Rm, t = T[:3, :3], T[:3, 3]
                if mode == 'matrix':
                    r = Rm.reshape(-1)
                    rot_str = ",".join(self._fmt(v, 3) for v in r)
                    trans_str = ",".join(self._fmt(v, 3) for v in t)
                    out.append(f"{prefix}R:[{rot_str}] T:[{trans_str}]")
                else:
                    q = R.from_matrix(Rm).as_quat()  # x,y,z,w
                    quat_str = ",".join(self._fmt(v, 3) for v in q)
                    trans_str = ",".join(self._fmt(v, 3) for v in t)
                    out.append(f"{prefix}Q:[{quat_str}] T:[{trans_str}]")
            return out

        # description mode
        from cambrianp.camera_trajectory_utils import CameraExtrinsicToText
        textifier = CameraExtrinsicToText(
            mode='description',
            reference_frame=self.reference_frame,
            text_strategy=self.text_strategy,
        )
        blob = textifier.process_extrinsics(E44)

        pair_re   = r'Frame\s+(\d+)\s*to\s*Frame\s+(\d+)\s*:\s*([^|]+)'
        single_re = r'Frame\s+(\d+)\s*:\s*([^|]+)'

        # Key by the *target* frame index i (the right index b)
        desc_by_i = {}
        for a_str, b_str, desc in re.findall(pair_re, blob, flags=re.IGNORECASE):
            try:
                b = int(b_str)
                desc_by_i[b] = desc.strip()
            except Exception:
                pass
        for i_str, desc in re.findall(single_re, blob, flags=re.IGNORECASE):
            try:
                idx = int(i_str)
                desc_by_i[idx] = desc.strip()
            except Exception:
                pass

        for i in range(N):
            if i == 0:
                out.append("Frame 0: Starting position")
            else:
                desc = desc_by_i.get(i, 'Hold')
                if self.reference_frame == 'first':
                    out.append(f"Frame 0 to Frame {i}: {desc}")
                else:
                    out.append(f"Frame {i-1} to Frame {i}: {desc}")
        return out

    def extract_camera_positions(self, extrinsics):
        E44 = self._normalize_extrinsics_44(extrinsics)
        positions = []
        directions = []
        for E in E44:
            cam_to_world = np.linalg.inv(E)
            pos = cam_to_world[:3, 3]
            direction = cam_to_world[:3, :3] @ np.array([0, 0, -1], dtype=np.float32)
            positions.append(pos)
            directions.append(direction)
        return np.array(positions), np.array(directions)

    def _create_trajectory_line(self, positions, num_points_per_segment=20):
        if len(positions) < 2:
            return positions
        traj = []
        for i in range(len(positions) - 1):
            start, end = positions[i], positions[i + 1]
            t = np.linspace(0, 1, num_points_per_segment)[:, None]
            traj.append(start + t * (end - start))
        traj.append(positions[-1:])
        return np.vstack(traj)

    def log_to_wandb(
        self,
        gt_extrinsics,
        pred_extrinsics,
        depths=None,
        intrinsics=None,
        images=None,
        world_points=None,
        point_masks=None,
        prefix="trajectory",
        frame_ids=None,        
        basenames=None,         
        scene_id=None,          
        max_cols=8,            
        Racc_15=None,
        Tacc_15=None,
        step=None,
    ):
        try:
            # Normalize extrinsics to [N,4,4]
            gt44 = self._normalize_extrinsics_44(gt_extrinsics)
            pred44 = self._normalize_extrinsics_44(pred_extrinsics)

            # Optional inputs (strip batch dim if present)
            if isinstance(depths, torch.Tensor) and depths.ndim == 4: depths = depths[0]
            if isinstance(intrinsics, torch.Tensor) and intrinsics.ndim == 4: intrinsics = intrinsics[0]
            if isinstance(images, torch.Tensor) and images.ndim == 5: images = images[0]
            if isinstance(world_points, torch.Tensor):
                world_points = world_points[0] if world_points.ndim in (4, 5) else world_points
            if isinstance(point_masks, torch.Tensor):
                point_masks = point_masks[0] if point_masks.ndim in (3, 4) else point_masks

            scene_points, scene_colors, boxes, _text_comparison_all = self.create_3d_object(
                gt_extrinsics=gt44,
                pred_extrinsics=pred44,
                depths=depths,
                intrinsics=intrinsics,
                images=images,
                world_points=world_points,
                point_masks=point_masks,
                max_points=50000
            )

            log_dict = {}

            if scene_points.size > 0 and wandb is not None:
                pts = scene_points.astype(np.float32)
                cols = scene_colors.astype(np.uint8)
                valid = np.isfinite(pts).all(axis=1)
                if valid.any():
                    pc = np.column_stack([pts[valid], cols[valid]])
                    log_dict[f"{prefix}/3d_scene"] = wandb.Object3D(pc)

            gt_inv0 = np.linalg.inv(gt44[0]); pred_inv0 = np.linalg.inv(pred44[0])
            gt_norm = [gt_inv0 @ m for m in gt44]
            pred_norm = [pred_inv0 @ m for m in pred44]
            gt_pos_n = np.array([m[:3, 3] for m in gt_norm])
            pred_pos_n = np.array([m[:3, 3] for m in pred_norm])

            # === Decide reporting mode and precompute text rows ===
            selected_mode = self.mode if self.mode in {"matrix", "quaternion", "description"} else "quaternion"
            gt_texts = self.extrinsics_to_text(gt44, mode=selected_mode)
            pred_texts = self.extrinsics_to_text(pred44, mode=selected_mode)

            # If using description mode, compute simple text-agreement metrics now
            desc_metrics = None
            if selected_mode == 'description':
                desc_metrics = self._compare_description_texts(gt_texts, pred_texts)

            fig = plt.figure(figsize=(16, 10))
            ax3d = fig.add_subplot(221, projection='3d')
            ax3d.plot(gt_pos_n[:,0], gt_pos_n[:,1], gt_pos_n[:,2], 'g-', linewidth=2, label='GT', marker='o')
            ax3d.plot(pred_pos_n[:,0], pred_pos_n[:,1], pred_pos_n[:,2], 'r-', linewidth=2, label='Pred', marker='s')
            ax3d.set_title('Normalized Trajectories'); ax3d.legend()
            ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')

            ax_xy = fig.add_subplot(222)
            ax_xy.plot(gt_pos_n[:,0], gt_pos_n[:,1], 'g-', linewidth=2, label='GT', marker='o')
            ax_xy.plot(pred_pos_n[:,0], pred_pos_n[:,1], 'r-', linewidth=2, label='Pred', marker='s')
            ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y'); ax_xy.set_title('Top-Down'); ax_xy.axis('equal'); ax_xy.grid(True, alpha=0.3); ax_xy.legend()

            ax_xz = fig.add_subplot(223)
            ax_xz.plot(gt_pos_n[:,0], gt_pos_n[:,2], 'g-', linewidth=2, label='GT', marker='o')
            ax_xz.plot(pred_pos_n[:,0], pred_pos_n[:,2], 'r-', linewidth=2, label='Pred', marker='s')
            ax_xz.set_xlabel('X'); ax_xz.set_ylabel('Z'); ax_xz.set_title('Side'); ax_xz.axis('equal'); ax_xz.grid(True, alpha=0.3); ax_xz.legend()

            # Aggregate errors (no per-frame plotting)
            # Position-based:
            gt_pos, _ = self.extract_camera_positions(gt44)
            pred_pos, _ = self.extract_camera_positions(pred44)
            ate = float(np.linalg.norm(gt_pos - pred_pos, axis=1).mean())
            if len(gt_pos) > 1:
                rpe = float(np.linalg.norm(np.diff(gt_pos, axis=0) - np.diff(pred_pos, axis=0), axis=1).mean())
            else:
                rpe = float('nan')

            # Rotation-based (mean & max only)
            rot_errors = []
            for i in range(min(len(gt_norm), len(pred_norm))):
                R_err = gt_norm[i][:3, :3].T @ pred_norm[i][:3, :3]
                ang = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
                rot_errors.append(np.degrees(ang))
            mean_rot_err = float(np.mean(rot_errors)) if rot_errors else float('nan')
            max_rot_err = float(np.max(rot_errors)) if rot_errors else float('nan')

            ax_txt = fig.add_subplot(224)
            ax_txt.axis('off')
            summary = (
                f"ATE (mean position error): {ate:.4f}\n"
                f"RPE (mean delta error):    {rpe:.4f}\n"
                f"Mean rotation error:       {mean_rot_err:.2f}°\n"
                f"Max rotation error:        {max_rot_err:.2f}°"
            )
            if Racc_15 is not None and Tacc_15 is not None:
                summary += (
                    f"\n\nAccuracy (consecutive pairs):\n"
                    f"Racc_15:                   {Racc_15:.2%}\n"
                    f"Tacc_15:                   {Tacc_15:.2%}"
                )
            
            if desc_metrics is not None:
                summary += (
                    f"\n\nText (description) agreement:\n"
                    f"Token F1 (mean):           {desc_metrics['desc_token_f1_mean']:.3f}\n"
                    f"Exact token set match:     {desc_metrics['desc_exact_match_rate']:.3f}\n"
                    f"Hold match rate:           {desc_metrics['desc_hold_match_rate']:.3f}"
                )

            ax_txt.text(0.0, 0.9, "Metrics Summary", fontsize=12, fontweight="bold", va="top")
            ax_txt.text(0.0, 0.75, summary, fontsize=11, va="top", family="monospace")
            
            plt.tight_layout()
            if wandb is not None:
                log_dict[f"{prefix}/trajectory_analysis"] = wandb.Image(fig)
            plt.close(fig)

            # Scalars only (no per-frame logs):
            if desc_metrics is not None:
                for k, v in desc_metrics.items():
                    log_dict[f"{prefix}/{k}"] = v
            log_dict[f"{prefix}/ate"] = ate
            log_dict[f"{prefix}/rpe"] = rpe
            log_dict[f"{prefix}/mean_rotation_error_deg"] = mean_rot_err
            log_dict[f"{prefix}/max_rotation_error_deg"] = max_rot_err

            # === Pose comparison: ONLY the selected mode, as a readable diagram ===
            cmp_fig = self._make_pose_comparison_figure(gt_texts, pred_texts, selected_mode)
            if wandb is not None:
                log_dict[f"{prefix}/poses_{selected_mode}"] = wandb.Image(cmp_fig)
            plt.close(cmp_fig)
            
            images_hwc = self._prep_images_hwc_uint8(images)  
            cmp_rgb_fig = self._make_pose_diagram_figure(
                images_hwc, gt_texts, pred_texts, selected_mode,
                frame_ids=frame_ids, basenames=basenames, scene_id=scene_id, max_cols=max_cols
            )
            if wandb is not None:
                log_dict[f"{prefix}/poses_{selected_mode}_rgb"] = wandb.Image(cmp_rgb_fig)
            plt.close(cmp_rgb_fig)

            # Final log
            if wandb is not None and log_dict:
                wandb.log(log_dict, step=step)

            # Return only the chosen mode for callers that consume text
            return {
                f'gt_{selected_mode}': gt_texts,
                f'pred_{selected_mode}': pred_texts,
            }

        except Exception:
            import traceback; traceback.print_exc()
            try:
                return {
                    'gt_quaternion': self.extrinsics_to_text(gt_extrinsics, mode='quaternion'),
                    'pred_quaternion': self.extrinsics_to_text(pred_extrinsics, mode='quaternion'),
                }
            except Exception:
                return {}
            
    def create_3d_object(
        self,
        gt_extrinsics,
        pred_extrinsics,
        depths=None,
        intrinsics=None,
        images=None,
        world_points=None,
        point_masks=None,
        max_points=10000
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict], Dict]:
        # 1) Camera positions (for trajectories & frustums)
        gt_pos, gt_dir = self.extract_camera_positions(gt_extrinsics)
        pred_pos, pred_dir = self.extract_camera_positions(pred_extrinsics)

        all_points, all_colors = [], []

        # 2) PRIMARY: Reconstruct scene from depth + K + (world->cam) extrinsics
        if depths is not None and intrinsics is not None:
            depths_np = self._to_numpy(depths)
            intrinsics_np = self._to_numpy(intrinsics)
            extrinsics_np = self._to_numpy(gt_extrinsics)

            if extrinsics_np.ndim == 4:  # [B,N, ...] -> take first if batched
                extrinsics_np = extrinsics_np[0]
            if extrinsics_np.shape[-2:] == (4, 4):
                extr_for_recon = extrinsics_np[:, :3, :]
            else:
                extr_for_recon = extrinsics_np

            images_np = None
            if images is not None:
                images_np = self._to_numpy(images)
                if images_np.ndim == 4 and images_np.shape[1] == 3:  # [N,C,H,W] -> [N,H,W,C]
                    images_np = np.transpose(images_np, (0, 2, 3, 1))

            scene_pts, scene_cols = self.reconstruct_scene_from_depth(
                depths_np, intrinsics_np, extr_for_recon, images_np, max_points=max_points // 2
            )
            if len(scene_pts) > 0:
                all_points.append(scene_pts.astype(np.float32))
                all_colors.append(scene_cols.astype(np.uint8))

        # 3) FALLBACK: precomputed world_points (+ mask)
        elif world_points is not None:
            wp = self._to_numpy(world_points)
            if wp.ndim > 2:
                wp = wp.reshape(-1, 3)
            if point_masks is not None:
                pm = self._to_numpy(point_masks, dtype='bool').reshape(-1)
                wp = wp[pm]
            valid = np.isfinite(wp).all(axis=1) & (np.linalg.norm(wp, axis=1) < 100)
            wp = wp[valid]
            if len(wp) > max_points // 2:
                idx = np.random.choice(len(wp), max_points // 2, replace=False)
                wp = wp[idx]
            if len(wp) > 0:
                all_points.append(wp.astype(np.float32))
                all_colors.append(np.ones((len(wp), 3), dtype=np.uint8) * 180)

        # 4) Trajectories (dense) + camera centers
        def _traj_dense(positions, n=20):
            if len(positions) < 2:
                return positions
            segs = []
            for i in range(len(positions) - 1):
                a, b = positions[i], positions[i + 1]
                t = np.linspace(0, 1, n)[:, None]
                segs.append(a + t * (b - a))
            segs.append(positions[-1:])
            return np.vstack(segs)

        gt_traj = _traj_dense(gt_pos, 20)
        pred_traj = _traj_dense(pred_pos, 20)

        all_points += [gt_traj, pred_traj, gt_pos, pred_pos]
        all_colors += [
            np.tile([0, 255, 0], (len(gt_traj), 1)),
            np.tile([255, 0, 0], (len(pred_traj), 1)),
            np.tile([0, 200, 0], (len(gt_pos), 1)),
            np.tile([255, 50, 50], (len(pred_pos), 1)),
        ]

        scene_points = np.vstack(all_points) if all_points else np.zeros((0, 3), dtype=np.float32)
        scene_colors = np.vstack(all_colors).astype(np.uint8) if all_colors else np.zeros((0, 3), dtype=np.uint8)

        valid = np.isfinite(scene_points).all(axis=1)
        scene_points = scene_points[valid].astype(np.float32)
        scene_colors = scene_colors[valid].astype(np.uint8)

        # 5) Frustum boxes
        boxes = []
        for i, (p, d) in enumerate(zip(gt_pos, gt_dir)):
            fr = self._create_camera_frustum(p, d, scale=0.2)
            boxes.append({"corners": fr.tolist(), "label": f"GT_F{i}", "color": [0, 255, 0], "score": 1.0})
        for i, (p, d) in enumerate(zip(pred_pos, pred_dir)):
            fr = self._create_camera_frustum(p, d, scale=0.2)
            boxes.append({"corners": fr.tolist(), "label": f"Pred_F{i}", "color": [255, 0, 0], "score": 1.0})

        # 6) Text snapshots for side-by-side (unchanged)
        text_comparison = {
            'gt_matrix': self.extrinsics_to_text(gt_extrinsics, mode='matrix'),
            'gt_quaternion': self.extrinsics_to_text(gt_extrinsics, mode='quaternion'),
            'gt_description': self.extrinsics_to_text(gt_extrinsics, mode='description'),
            'pred_matrix': self.extrinsics_to_text(pred_extrinsics, mode='matrix'),
            'pred_quaternion': self.extrinsics_to_text(pred_extrinsics, mode='quaternion'),
            'pred_description': self.extrinsics_to_text(pred_extrinsics, mode='description'),
        }

        return scene_points, scene_colors, boxes, text_comparison

    def _depth_to_pointcloud(
        self,
        depth_map: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image: np.ndarray = None,
        H: int = None,
        W: int = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if H is None or W is None:
            H, W = depth_map.shape

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.reshape(-1)
        v = v.reshape(-1)
        depth = depth_map.reshape(-1)

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        Rm = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        points_world = (Rm.T @ (points_cam - t).T).T  # world = R^T (cam - t)

        colors = None
        if image is not None:
            if image.ndim == 3 and image.shape[0] == 3:  # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
            colors = image.reshape(-1, 3)
            colors = (colors * 255).astype(np.uint8) if colors.max() <= 1.0 else colors.astype(np.uint8)

        return points_world, colors

    

    def reconstruct_scene_from_depth(
        self,
        depths: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        images: np.ndarray = None,
        max_points: int = 10000,
        depth_threshold: Tuple[float, float] = (0.1, 10.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = len(depths)
        all_points, all_colors = [], []

        for i in range(N):
            depth_map = depths[i]
            K = intrinsics[i]
            E = extrinsics[i]  # world->cam, shape (3,4) or (4,4)
            if E.shape[-2:] == (4, 4):
                E = E[:3, :4]
            H, W = depth_map.shape

            img = None
            if images is not None:
                img = images[i]

            pts, cols = self._depth_to_pointcloud(depth_map, K, E, img, H, W)

            dflat = depth_map.reshape(-1)
            valid = (dflat > depth_threshold[0]) & (dflat < depth_threshold[1])
            valid &= np.isfinite(pts).all(axis=1)
            pts = pts[valid]
            cols = (cols[valid] if cols is not None else np.ones((len(pts), 3), dtype=np.uint8) * 180)

            all_points.append(pts)
            all_colors.append(cols)

        if not all_points:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        scene_points = np.vstack(all_points)
        scene_colors = np.vstack(all_colors)

        if len(scene_points) > max_points:
            idx = np.random.choice(len(scene_points), max_points, replace=False)
            scene_points = scene_points[idx]
            scene_colors = scene_colors[idx]

        return scene_points, scene_colors
    
    def _make_pose_comparison_figure(self, gt_rows, pred_rows, mode: str):
        n = min(len(gt_rows), len(pred_rows))
        # Scale height with #frames for readability
        fig_h = max(4.0, 0.45 * n)
        fig = plt.figure(figsize=(12, fig_h))
        ax = fig.add_subplot(111)
        ax.axis("off")

        # Build table data
        cell_text = [[f"{i}", gt_rows[i], pred_rows[i]] for i in range(n)]
        col_labels = ["Frame", "Ground Truth", "Predicted"]

        tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                    cellLoc="left", loc="upper left")
        # Styling
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.2)

        ax.set_title(f"Pose Comparison ({mode})", pad=10)
        plt.tight_layout()
        return fig
    
    def _create_camera_frustum(self, position, direction, up=None, scale=0.3):
        if up is None:
            up = np.array([0, 1, 0], dtype=np.float32)

        position = np.asarray(position, dtype=np.float32)
        direction = np.asarray(direction, dtype=np.float32)

        # Orthonormal basis
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        right = np.cross(direction, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, direction)
        up = up / (np.linalg.norm(up) + 1e-8)

        apex = position
        center = position + direction * scale
        half_w = scale * 0.4
        half_h = scale * 0.3

        corners = np.array([
            center + right * half_w + up * half_h,
            center + right * half_w - up * half_h,
            center - right * half_w - up * half_h,
            center - right * half_w + up * half_h,
        ], dtype=np.float32)

        return np.vstack([apex.reshape(1, 3), corners])

    def _prep_images_hwc_uint8(self, images):
        if images is None:
            return None
        arr = self._to_numpy(images)
        if arr.ndim == 4 and arr.shape[1] == 3:         # [N,C,H,W] -> [N,H,W,C]
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.ndim != 4 or arr.shape[-1] != 3:
            return None
        if arr.dtype != np.uint8:
            arr = (arr * 255.0).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return arr

    def _make_pose_diagram_figure(
        self,
        images_hwc,
        gt_rows,
        pred_rows,
        mode: str,
        *,
        frame_ids=None,
        basenames=None,
        scene_id=None,
        max_cols=8,
        max_txt=140
    ):
        import os
        n_txt = min(len(gt_rows), len(pred_rows))
        n_img = images_hwc.shape[0] if images_hwc is not None else n_txt
        n = int(max(1, min(n_txt, n_img)))

        cols = max(1, int(max_cols))
        chunks = [(s, min(s + cols, n)) for s in range(0, n, cols)]

        # Figure size scales with number of chunks (pages) and cols
        fig_w = min(2.2 * min(cols, n), 28.0)
        fig_h = 6.2 * len(chunks)
        fig = plt.figure(figsize=(fig_w, fig_h))
        outer = fig.add_gridspec(len(chunks), 1, hspace=0.45)

        def _short(s: str, lim: int) -> str:
            s = str(s)
            return s if len(s) <= lim else (s[:lim - 1] + "…")

        for row, (s, e) in enumerate(chunks):
            gs = outer[row].subgridspec(4, e - s, height_ratios=[0.6, 3.0, 1.7, 1.7], hspace=0.22, wspace=0.05)

            # Row 0: indices + ids + basenames (truncated)
            for c in range(s, e):
                ax = fig.add_subplot(gs[0, c - s]); ax.axis('off')
                idx_str = f"F{c}"
                if frame_ids is not None and c < len(frame_ids):
                    try:
                        idx_str += f"\nID:{int(frame_ids[c])}"
                    except Exception:
                        idx_str += f"\nID:{frame_ids[c]}"
                base_str = ""
                if basenames is not None and c < len(basenames):
                    base = basenames[c]
                    base = os.path.basename(base) if isinstance(base, str) else str(base)
                    base_str = _short(base, 18)
                ax.text(0.5, 0.82, idx_str, ha='center', va='top', fontsize=9, fontweight='bold', transform=ax.transAxes)
                if base_str:
                    ax.text(0.5, 0.06, base_str, ha='center', va='bottom', fontsize=7, transform=ax.transAxes)

            # Row 1: RGB
            for c in range(s, e):
                ax = fig.add_subplot(gs[1, c - s]); ax.axis('off')
                if images_hwc is not None and c < len(images_hwc):
                    ax.imshow(images_hwc[c])
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=8)

            # Row 2: Pred (truncate to avoid overflow)
            for c in range(s, e):
                ax = fig.add_subplot(gs[2, c - s]); ax.axis('off')
                if c == s:
                    ax.text(0.02, 1.03, "Pred", ha='left', va='bottom', fontsize=9, fontweight='bold', transform=ax.transAxes)
                txt = pred_rows[c] if c < len(pred_rows) else ""
                ax.text(0.02, 0.95, _short(txt, max_txt), ha='left', va='top', wrap=True, fontsize=8, family='monospace',
                        transform=ax.transAxes)

            # Row 3: GT
            for c in range(s, e):
                ax = fig.add_subplot(gs[3, c - s]); ax.axis('off')
                if c == s:
                    ax.text(0.02, 1.03, "GT", ha='left', va='bottom', fontsize=9, fontweight='bold', transform=ax.transAxes)
                txt = gt_rows[c] if c < len(gt_rows) else ""
                ax.text(0.02, 0.95, _short(txt, max_txt), ha='left', va='top', wrap=True, fontsize=8, family='monospace',
                        transform=ax.transAxes)

        title = f"Pose Diagram ({mode})"
        if scene_id:
            title += f" — Scene: {scene_id}"
        fig.suptitle(title, y=0.995, fontsize=11, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def _prep_images_hwc_uint8(self, images):
        if images is None:
            return None
        arr = self._to_numpy(images)
        if arr.ndim == 4 and arr.shape[1] == 3:  # [N,C,H,W] -> [N,H,W,C]
            arr = np.transpose(arr, (0, 2, 3, 1))
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return arr

    def _desc_line_to_tokens(self, line: str):
        """
        Map a description line to an order-invariant token set:
        movement: move_left/right/up/down/forward/backward
        rotation: turn_left/right, look_up/down
        roll:     lean_left/right  (only if present; ignored otherwise)
        hold:     hold
        """
        if not isinstance(line, str):
            return set()
        # keep only the part after ':'
        desc = line.split(':', 1)[-1].strip().lower()

        # trivial cases
        if desc in ("hold", "starting position"):
            return {"hold"}

        toks = set()

        # movement (metric/simple/professional synonyms)
        if re.search(r'\b(move\s+)?right\b', desc) or 'truck right' in desc or re.search(r'\bright\s+[\d.]+m\b', desc):
            toks.add('move_right')
        if re.search(r'\b(move\s+)?left\b', desc) or 'truck left' in desc or re.search(r'\bleft\s+[\d.]+m\b', desc):
            toks.add('move_left')
        if re.search(r'\b(move\s+)?up\b', desc) or 'pedestal up' in desc or re.search(r'\bup\s+[\d.]+m\b', desc):
            toks.add('move_up')
        if re.search(r'\b(move\s+)?down\b', desc) or 'pedestal down' in desc or re.search(r'\bdown\s+[\d.]+m\b', desc):
            toks.add('move_down')
        if 'move forward' in desc or 'dolly in' in desc:
            toks.add('move_forward')
        if 'move backward' in desc or 'dolly out' in desc or 'move back' in desc:
            toks.add('move_backward')

        # yaw
        if 'turn left' in desc or 'pan left' in desc:
            toks.add('turn_left')
        if 'turn right' in desc or 'pan right' in desc:
            toks.add('turn_right')

        # pitch
        if 'look up' in desc or 'tilt up' in desc:
            toks.add('look_up')
        if 'look down' in desc or 'tilt down' in desc:
            toks.add('look_down')

        # roll (optional terms)
        if 'lean right' in desc or 'clockwise dutch angle' in desc or re.search(r'\broll\s*[+]?[\d.]+', desc):
            toks.add('lean_right')
        if 'lean left' in desc or 'counterclockwise dutch angle' in desc or re.search(r'\broll\s*-?[\d.]+', desc):
            # crude split of +/-, but above line already catches plus; this catches negatives
            toks.add('lean_left')

        if not toks:
            # If nothing detected, treat as hold to avoid punishing empty pairs
            return {"hold"}
        return toks

    def _compare_description_texts(self, gt_rows, pred_rows):
        """
        Compare per-pair rows (skip row 0 baseline). Returns aggregate text metrics.
        """
        n = min(len(gt_rows), len(pred_rows))
        if n <= 1:
            return {'desc_token_f1_mean': float('nan'),
                    'desc_exact_match_rate': float('nan'),
                    'desc_hold_match_rate': float('nan')}

        f1s, exacts, holds = [], [], []
        for i in range(1, n):
            gt = self._desc_line_to_tokens(gt_rows[i])
            pr = self._desc_line_to_tokens(pred_rows[i])

            inter = len(gt & pr)
            p = inter / max(len(pr), 1)
            r = inter / max(len(gt), 1)
            f1 = 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)

            f1s.append(f1)
            exacts.append(1.0 if gt == pr else 0.0)
            holds.append(1.0 if (gt == {'hold'} and pr == {'hold'}) else 0.0)

        return {
            'desc_token_f1_mean': float(np.mean(f1s)) if f1s else float('nan'),
            'desc_exact_match_rate': float(np.mean(exacts)) if exacts else float('nan'),
            'desc_hold_match_rate': float(np.mean(holds)) if holds else float('nan'),
        }

       
def get_camera_system_prompt(
    task_mode: str,
    reference_frame: str,
    supervision_type: str = 'quaternion',
    text_strategy: str = 'metric',
    *,
    use_cam_as_input: bool = False,   
    pairwise: bool = False     
):
    if use_cam_as_input:
        if task_mode == "camera_trajectory":
            # Consume hints but still OUTPUT a trajectory (unless your dataset expects only answers)
            label = supervision_type if supervision_type in ("quaternion", "matrix") else f"natural language ({text_strategy})"
            return (
                "You will see frames interleaved with short camera transition hints. "
                f"Use them to reason about motion, then output the full trajectory strictly in {label} "
                f"relative to the {reference_frame} frame. Do not repeat the hints verbatim."
            )
        else:
            # VQA with motion hints as context
            return (
                "You will receive frames interleaved with short camera transition hints. "
                "Use these as auxiliary context when answering. Do not repeat the hints."
            )

    if task_mode == "camera_trajectory":
        if supervision_type in ("quaternion", "matrix"):
            return (
                f"You are a camera trajectory analyzer. Use {supervision_type} relative to the "
                f"{reference_frame} frame. Format F0..Fn exactly."
            )
        else:
            if pairwise:
                return (
                    f"You are a camera trajectory analyzer. Use natural language ({text_strategy}) "
                    "to describe consecutive frame motions as 'Frame i to Frame i+1: ...' joined by '|'."
                )
            else:
                return (
                    f"You are a camera trajectory analyzer. Use natural language ({text_strategy}) "
                    f"relative to the {reference_frame} frame. Write 'Frame k: ...' joined by '|'."
                )
    else:
        if supervision_type in ("quaternion", "matrix"):
            return (
                f"When answering, first output camera trajectory in {supervision_type} relative to the "
                f"{reference_frame} frame, then write 'Answer:' and the answer."
            )
        else:
            return (
                f"When answering, first describe the camera trajectory in natural language ({text_strategy}) "
                f"relative to the {reference_frame} frame, then write 'Answer:' and the answer."
            )