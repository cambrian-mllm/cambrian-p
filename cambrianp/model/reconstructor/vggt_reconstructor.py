import torch
import torch.nn as nn
import math

from typing import Union, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast

from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.dpt_head_rms import DPTHeadRMS
from vggt.loss.loss import MultitaskLoss

from cambrianp.utils import rank0_print
from cambrianp.model.multimodal_projector.builder import build_projector


class VGGTReconstructor(nn.Module):
    def __init__(self,  config):
        super().__init__()
        
        self.config = config
        pretrained_model_path = config.rec_model_path
        self.num_intermediate_layers = config.num_intermediate_layers
        self.enable_camera = config.enable_camera
        self.enable_point = config.enable_point
        self.enable_depth = config.enable_depth
        
        self.camera_tokens_num = self.config.token_num
        self.camera_tokens_mode = self.config.camera_tokens_mode
        self.camera_tokens_place = getattr(self.config, "camera_tokens_place", "prepend_to_frame")
        
        self.query_mode = self.config.query_mode
        self.query_loss_frame_idx = self.config.query_loss_frame_idx
        
        self.patch_start_idx = self.config.rec_depth_head_patch_start_idx
        self.patch_size = int(math.sqrt(self.config.mm_img_tok_num  ))
        
        rank0_print("token_num: {}, camera_tokens_mode: {}, query_mode: {}, camera_tokens_place: {}".format(
            self.camera_tokens_num, self.camera_tokens_mode, self.query_mode, self.camera_tokens_place
        ))
        
        self._init_camera_tokens()
        self._init_loss()
        
        if not self.enable_depth:
            # the depth head is not used, so we only need one layer
            self.num_intermediate_layers = 1
        
        self.projector_list = nn.ModuleList([
            build_projector(
                projector_type=config.rec_projector_type,
                mm_hidden_size=config.hidden_size,
                hidden_size=config.rec_embed_dim,
                patch_start_idx=self.patch_start_idx,
                patch_size=self.patch_size
            )
            for _ in range(self.num_intermediate_layers)
        ])
    
        if self.enable_camera:
            # the camera head of VGGT only take the last layer's features
            self.camera_head = CameraHead(
                dim_in=config.rec_embed_dim,
                trunk_depth=getattr(config, "rec_camera_trunk_depth", 4),
                is_causal=getattr(config, "rec_camera_causal_attn", False),
            )
            rank0_print(f"\033[31mCameraHead num_params: {sum(p.numel() for p in self.camera_head.parameters()) / 1e6:.2f}M\033[0m")
        else:
            self.camera_head = None
        
        dpt_head_type = getattr(config, "rec_dpt_head_type", "dpt_head")
        if dpt_head_type == "dpt_head":
            self.dpt_head = DPTHead
        elif dpt_head_type == "dpt_head_rms":
            self.dpt_head = DPTHeadRMS
        else:
            raise ValueError(f"Invalid DPT head type: {dpt_head_type}")
        
        if self.enable_point:
            self.point_head = self.dpt_head(
                dim_in=config.rec_embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1",
                intermediate_layer_idx=list(range(self.num_intermediate_layers)), patch_size=self.patch_size
            )
        else:
            self.point_head = None
        
        if self.enable_depth:
            self.depth_head = self.dpt_head(
                dim_in=config.rec_embed_dim, output_dim=2, activation="exp", conf_activation="expp1",
                intermediate_layer_idx=list(range(self.num_intermediate_layers)), patch_size=self.patch_size
            )
            rank0_print(f"VGGTReconstructor Depth Head", self.depth_head)
        else:
            self.depth_head = None
        
        self.apply(self._init_weights)
        
        if pretrained_model_path != 'None':
            self._load_pretrained_model(pretrained_model_path)
            rank0_print(f"VGGTReconstructor: loaded pretrained model from {pretrained_model_path}")
        else:
            # init the model with random weights
            rank0_print("VGGTReconstructor: no pretrained model loaded")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_camera_tokens(self):
        # change this from 896 -> 3584 for 7b
        hidd_dim = self.config.hidden_size

        # Create parameter with dtype set inside the constructor to preserve nn.Parameter type
        # Use zeros since nn.init.normal_ will overwrite values anyway
        self.camera_tokens = nn.Parameter(
            torch.zeros(self.camera_tokens_num, hidd_dim, dtype=torch.bfloat16)
        )
        
        nn.init.normal_(self.camera_tokens, std=1e-6)
        
    def _init_loss(self):
        camera_loss_config = {
            'weight': self.config.rec_camera_loss_weight,
            'loss_type': self.config.rec_camera_loss_type,
            'components': getattr(self.config, 'rec_camera_loss_components', 'both'),
            'use_point_masks': getattr(self.config, 'use_point_masks', True),
            'use_scale_alignment': getattr(self.config, 'use_scale_alignment', False),
            'use_trajectory_balance': getattr(self.config, 'use_trajectory_balance', False),
            'weight_trans': getattr(self.config, 'rec_camera_loss_weight_trans', 1.0),
            'weight_rot': getattr(self.config, 'rec_camera_loss_weight_rot', 1.0),
            'weight_focal': getattr(self.config, 'rec_camera_loss_weight_focal', 0.5),
            'rescale_pred_not_gt': getattr(self.config, 'rescale_pred_not_gt', False),
        }
        depth_loss_config = {
            'weight': self.config.rec_depth_loss_weight,
            'gradient_loss_fn': self.config.rec_depth_gradient_loss_fn,
            'valid_range': self.config.rec_depth_valid_range,
            'use_point_masks': getattr(self.config, 'use_point_masks', True),
        }
        point_loss_config = None
        
        self.loss = MultitaskLoss(
            camera=camera_loss_config, depth=depth_loss_config, point=point_loss_config
        )
    
    def _load_pretrained_model(self, pretrained_model_path: str):
        model_dict = torch.load(pretrained_model_path, weights_only=True)
        
        for key in model_dict:
            if isinstance(model_dict[key], torch.Tensor):
                model_dict[key] = model_dict[key].to(torch.bfloat16)
        
        self.load_state_dict(model_dict, strict=False)

    def forward_projector(
        self,
        outputs: Union[Tuple, CausalLMOutputWithPast],
        info_dict_list: List[dict],
    ) -> List[torch.Tensor]:
        # project the features and extract the features for reconstruction
        if self.camera_tokens_mode == "camera_tokens":
            if self.camera_tokens_place == "between_qa":
                rec_feats_list = self.prepare_camera_tokens_for_per_frame_between_qa(
                    info_dict_list, outputs
                )
            else:
                rec_feats_list = self.prepare_camera_tokens_for_per_frame(
                    info_dict_list, outputs
                )
        elif self.camera_tokens_mode == "query":
            rec_feats_list = self.prepare_camera_tokens_for_query(
                info_dict_list, outputs
            )
        else:
            raise ValueError(f"Invalid camera tokens mode: {self.camera_tokens_mode}")
        
        return rec_feats_list
    
    def forward_rec_heads(
        self,
        rec_feats_list: List[torch.Tensor],
        info_dict_list: List[dict],
        batch: dict,
    ) -> dict:
        images = batch["images"]
        
        predictions = {}
        
        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(rec_feats_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    rec_feats_list, images=images, patch_start_idx=self.patch_start_idx, frames_chunk_size=32
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
                
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    rec_feats_list, images=images, patch_start_idx=self.patch_start_idx, frames_chunk_size=32
                )
                predictions["pts3d"] = pts3d
                predictions["pts3d_conf"] = pts3d_conf
                
        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference
            
        return predictions

    def forward(
        self,
        outputs: Union[Tuple, CausalLMOutputWithPast],
        info_dict_list: List[dict],
        batch: dict,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        rec_feats_list = self.forward_projector(outputs, info_dict_list)
        
        # forward the reconstruction heads to compute the loss
        predictions = self.forward_rec_heads(rec_feats_list, info_dict_list, batch)
        
        if not self.training:
            return predictions
        
        # calculate the loss for reconstruction
        loss_dict = self.forward_loss(predictions, batch)
        return loss_dict, predictions
    
    def forward_loss(
        self,
        predictions: dict,
        batch: dict,
    ) -> dict:
        
        loss_dict = self.loss(predictions, batch)
        
        return loss_dict
    
    def inject_rec_tokens_to_llm(
        self,
        images: torch.Tensor,
        batch_idx: int,
        cur_labels: torch.Tensor,
        cur_new_input_embeds: List[torch.Tensor],
        cur_new_labels: List[torch.Tensor],
        IGNORE_INDEX: int,
        bp_vqa: bool = True,
        cam_segment_idx: int = -1,   
        cam_position_in_segment: int = -1,  
    ) -> Tuple[dict, List[torch.Tensor], List[torch.Tensor]]:

        if self.camera_tokens_mode == "camera_tokens":
            num_frames = images[batch_idx].shape[0]
            feat_dim = cur_new_input_embeds[-1].shape[-1]
            token_num = self.camera_tokens_num
            im_features = cur_new_input_embeds[1]
            im_features = im_features.view(num_frames, -1, feat_dim)
            
            if self.camera_tokens.device != im_features.device:
                self.camera_tokens = self.camera_tokens.to(im_features.device)
            
            if token_num == 2:
                camera_feat = torch.stack([self.camera_tokens[0]] + [self.camera_tokens[1]] * (num_frames - 1), dim=0)
            elif token_num == 3:
                camera_feat = torch.stack(
                    [self.camera_tokens[0]] + [self.camera_tokens[1]] * (num_frames - 2) + [self.camera_tokens[2]], 
                    dim=0
                )
            else:
                raise ValueError(f"Unexpected token_num: {token_num}")
            
            camera_seq = camera_feat
            camera_seq_labels = torch.full(
                (camera_seq.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype
            )
            
            if self.camera_tokens_place == "between_qa":
                if cam_segment_idx >= 0 and cam_segment_idx < len(cur_new_input_embeds):
                    # Typically cam_segment_idx would be 2 (the segment after the image_path features)
                    post_labels = cur_new_labels[cam_segment_idx]
                    post_embeds = cur_new_input_embeds[cam_segment_idx]

                    q_text_emb = post_embeds[:cam_position_in_segment]
                    q_text_lbl = post_labels[:cam_position_in_segment]
        
                    a_text_emb = post_embeds[cam_position_in_segment:]
                    a_text_lbl = post_labels[cam_position_in_segment:]
                    
                    if not bp_vqa:  # Reconstruction mode
                        # Set all labels to IGNORE_INDEX for rec mode
                        q_text_lbl = torch.full_like(q_text_lbl, IGNORE_INDEX)
                        a_text_lbl = torch.full_like(a_text_lbl, IGNORE_INDEX)
                    
                    #  [prompt, question, camera_tokens, answer]
                    new_embeds = []
                    new_labels = []
                    
                    # Add all segments before the one with CAM token
                    for i in range(cam_segment_idx):
                        new_embeds.append(cur_new_input_embeds[i])
                        new_labels.append(cur_new_labels[i])
                    
                    # Add the split segment with camera tokens inserted
                    new_embeds.append(q_text_emb)
                    new_labels.append(q_text_lbl)
                    new_embeds.append(camera_seq)
                    new_labels.append(camera_seq_labels)
                    
                    if a_text_emb.shape[0] > 0:
                        new_embeds.append(a_text_emb)
                        new_labels.append(a_text_lbl)
                    
                    for i in range(cam_segment_idx + 1, len(cur_new_input_embeds)):
                        new_embeds.append(cur_new_input_embeds[i])
                        new_labels.append(cur_new_labels[i])

                    info_dict = {
                        "pre_image_token_length": cur_new_input_embeds[0].shape[0],
                        "image_token_length": cur_new_input_embeds[1].shape[0],
                        "question_token_length": q_text_emb.shape[0],
                        "camera_token_count": camera_seq.shape[0],
                        "answer_token_length": a_text_emb.shape[0],
                        "num_frames": num_frames,
                        "camera_tokens_place": "between_qa",
                        "is_rec_only": not bp_vqa,
                    }
                    
                    cur_new_input_embeds[:] = new_embeds
                    cur_new_labels[:] = new_labels
                    
                else:
                    raise ValueError("camera token not found in the labels")
                    
            else:  # prepend_to_frame mode
                
                camera_feat = camera_seq.unsqueeze(1)
                
                # NOTE: the camera token is placed before the image token, to follow the order of the VGGT
                if self.camera_tokens_place == "prepend_to_frame":
                    im_feat_with_camera = torch.cat((camera_feat, im_features), dim=1).view(-1, feat_dim)
                elif self.camera_tokens_place == "append_to_frame":
                    im_feat_with_camera = torch.cat((im_features, camera_feat), dim=1).view(-1, feat_dim)
                else:
                    raise ValueError(f"Invalid camera tokens place, between qa mode needs rechecking: {self.camera_tokens_place}")
                
                # im_feat_with_camera = torch.cat((camera_feat, im_features), dim=1).view(-1, feat_dim)
                
                cur_new_input_embeds[1] = im_feat_with_camera
                cur_new_labels[1] = torch.full(
                    (im_feat_with_camera.shape[0],), IGNORE_INDEX,
                    device=cur_labels.device, dtype=cur_labels.dtype
                )
                
                if not bp_vqa and len(cur_new_labels) > 2:
                    cur_new_labels[2] = torch.full_like(
                        cur_new_labels[2], IGNORE_INDEX
                    )
                info_dict = {
                    "pre_image_token_length": cur_new_input_embeds[0].shape[0],
                    "image_token_length": cur_new_input_embeds[1].shape[0],
                    "post_image_token_length": cur_new_input_embeds[2].shape[0],
                    "num_frames": num_frames,
                    "camera_tokens_place": self.camera_tokens_place,
                    "is_rec_only": not bp_vqa,
                }
                
        elif self.camera_tokens_mode == "query":
            if self.query_mode == "query_after_image":
                insert_idx = 2  # [text before img, img, query, text after img]
            elif self.query_mode == "query_after_question":
                insert_idx = len(cur_new_input_embeds)  # [text before img, img, text after img, query]
            else:
                raise NotImplementedError("Incorrect Query mode")
            cur_new_input_embeds.insert(insert_idx, self.camera_tokens)

            cur_new_labels.insert(
                insert_idx,
                torch.full(
                    (self.camera_tokens.shape[0],), IGNORE_INDEX,
                    device=cur_labels.device, dtype=cur_labels.dtype
                )
            )
            info_dict = {
                "camera_token_count": self.camera_tokens.shape[0],
                "pre_image_token_length": cur_new_input_embeds[0].shape[0],
                "image_token_length": cur_new_input_embeds[1].shape[0],
                "post_image_token_length": cur_new_input_embeds[2].shape[0],
                "is_rec_only": not bp_vqa,
            }

        else:
            raise ValueError(f"Invalid camera tokens mode: {self.camera_tokens_mode}")

        return info_dict, cur_new_input_embeds, cur_new_labels
    
    def prepare_camera_tokens_for_per_frame(
        self,
        info_dict_list: List[dict],
        outputs: CausalLMOutputWithPast,
    ) -> List[torch.Tensor]:
        hidden_states_depth = len(outputs.hidden_states) - 1
        
        # the last layer have to be -1
        assert self.num_intermediate_layers >= 1
        
        intermediate_layer_idx = [
            hidden_states_depth * (i + 1) // self.num_intermediate_layers 
            for i in range(self.num_intermediate_layers - 1)
        ]
        intermediate_layer_idx.append(-1)
        
        rec_feats_list = []
        for layer_idx in intermediate_layer_idx:
            rec_feats_list.append(
                self.prepare_rec_tokens_for_single_layer(
                    info_dict_list, outputs.hidden_states, layer_idx=layer_idx, 
                    grid_size=self.patch_size, need_camera_token=True
                )
            )
        
        projected_rec_feats_list = []
        for i in range(len(rec_feats_list)):
            # frame-wise projection
            projected_rec_feats = self.projector_list[i](rec_feats_list[i])

            projected_rec_feats_list.append(projected_rec_feats)
        
        return projected_rec_feats_list

    def prepare_camera_tokens_for_query(self, info_dict_list: List[dict], outputs: CausalLMOutputWithPast):
        token_num = self.camera_tokens_num
        
        if self.query_mode == "query_after_image":
            learnable_tokens_list = []
            for i, info_dict in enumerate(info_dict_list):
                camera_token_start_idx = info_dict["pre_image_token_length"] + info_dict["image_token_length"] 
                # [text before img, img, query, text after img]
                learnable_tokens = outputs.hidden_states[-1][i, camera_token_start_idx:camera_token_start_idx + token_num] 
                learnable_tokens_list.append(learnable_tokens)
            learnable_tokens = torch.stack(learnable_tokens_list, dim=0)
        elif self.query_mode == "query_after_question":
            # [text before img, img, text after img, query]
            learnable_tokens = outputs.hidden_states[-1][:, -token_num:]
        else:
            raise NotImplementedError("Incorrect Query mode")
        
        learnable_tokens = self.projector_list[-1](learnable_tokens)
        
        return learnable_tokens
    
    @staticmethod
    def prepare_rec_tokens_for_single_layer(
        info_dict_list: List[dict],
        hidden_states: List[torch.Tensor],
        layer_idx: int = -1,
        grid_size: int = 14,
        need_camera_token: bool = True,
    ) -> torch.Tensor:
        learnable_tokens_list = []
        for i, info_dict in enumerate(info_dict_list):
            pre_image_token_length = info_dict["pre_image_token_length"]
            image_token_length = info_dict["image_token_length"]
            num_frame = info_dict["num_frames"]
        
            feat_dim = hidden_states[layer_idx].shape[-1]
            batch_size = hidden_states[layer_idx].shape[0]

            end_idx = image_token_length + pre_image_token_length
            learnable_tokens = hidden_states[layer_idx][i, pre_image_token_length:end_idx]
            learnable_tokens_list.append(learnable_tokens)

        learnable_tokens = torch.stack(learnable_tokens_list, dim=0)
        learnable_tokens = learnable_tokens.view(batch_size, num_frame, -1, feat_dim)
        
        token_per_frame = learnable_tokens.shape[-2]
        
        # Camera token is placed before the image token to match VGGT's order.
        # All samples in the batch share the same camera_tokens_place.
        if info_dict_list[0]["camera_tokens_place"] == "prepend_to_frame":
            camera_token_3r, image_tokens_3r = torch.split(learnable_tokens, [1, token_per_frame - 1], dim=2)
        elif info_dict_list[0]["camera_tokens_place"] == "append_to_frame":
            image_tokens_3r, camera_token_3r = torch.split(learnable_tokens, [token_per_frame - 1, 1], dim=2)
        else:
            raise ValueError(f"Invalid camera tokens place, between qa mode needs rechecking: {info_dict_list[0]['camera_tokens_place']}")
        # camera_token_3r, image_tokens_3r = torch.split(learnable_tokens, [1, token_per_frame - 1], dim=2)
        
        image_tokens_3r = image_tokens_3r.view(batch_size, num_frame, grid_size, grid_size + 1, feat_dim)
        image_tokens_3r = image_tokens_3r[..., :grid_size, :]
        image_tokens_3r = image_tokens_3r.reshape(batch_size, num_frame, grid_size*grid_size, feat_dim)
        
        if need_camera_token:
            combined_tokens_3r = torch.cat([camera_token_3r, image_tokens_3r], dim=2)
        else:
            combined_tokens_3r = image_tokens_3r
        
        return combined_tokens_3r
    
    def prepare_camera_tokens_for_per_frame_between_qa(
        self, 
        info_dict_list: List[dict], 
        outputs: CausalLMOutputWithPast
    ) -> List[torch.Tensor]:
        hidden_states_depth = len(outputs.hidden_states) - 1
        assert self.num_intermediate_layers >= 1
        
        intermediate_layer_idx = [
            hidden_states_depth * (i + 1) // self.num_intermediate_layers
            for i in range(self.num_intermediate_layers - 1)
        ]
        intermediate_layer_idx.append(-1)
        
        rec_feats_list = []
        for li, layer_idx in enumerate(intermediate_layer_idx):
            img_tokens = self._slice_image_tokens_without_cam(info_dict_list, outputs.hidden_states, layer_idx)
            cam_tokens = self._slice_camera_tokens_from_between_qa(info_dict_list, outputs.hidden_states, layer_idx)
            combined = torch.cat([cam_tokens, img_tokens], dim=2)
            rec_feats_list.append(combined)
        
        projected = []
        for i, x in enumerate(rec_feats_list):
            y = self.projector_list[i](x)
            projected.append(y)
        return projected

    def _slice_image_tokens_without_cam(self, info_dict_list, hidden_states, layer_idx):
        batch_feats = []
        grid = self.patch_size  # 14
        for i, info in enumerate(info_dict_list):
            pre = info["pre_image_token_length"]
            img_len = info["image_token_length"]
            num_frame = info["num_frames"]
            feat = hidden_states[layer_idx][i, pre:pre+img_len]
            C = feat.shape[-1]
            
            Tpf = feat.shape[0] // num_frame
            assert feat.shape[0] == num_frame * Tpf
            
            cols = Tpf // grid
            if cols == grid + 1:
                feat = feat.view(num_frame, grid, cols, C)
                feat = feat[:, :, :grid, :].reshape(num_frame, grid*grid, C)  # handle newline token
            elif cols == grid:
                feat = feat.view(num_frame, grid, cols, C)
                feat = feat[:, :, :grid, :].reshape(num_frame, grid*grid, C)
            else:
                raise ValueError(f"Invalid cols: {cols}")
            
            batch_feats.append(feat)
        
        out = torch.stack(batch_feats, dim=0)
        return out

    def _slice_camera_tokens_from_between_qa(self, info_dict_list, hidden_states, layer_idx):
        batch_cam = []
        for i, info in enumerate(info_dict_list):
            pre = info["pre_image_token_length"]
            img = info["image_token_length"]
            qlen = info["question_token_length"]
            clen = info["camera_token_count"]
            
            cam_start = pre + img + qlen
            cam_end = cam_start + clen
            
            cam_feat = hidden_states[layer_idx][i, cam_start:cam_end]
            cam_feat = cam_feat.unsqueeze(1)  # [S,1,C]
            batch_cam.append(cam_feat)
        
        out = torch.stack(batch_cam, dim=0)
        return out