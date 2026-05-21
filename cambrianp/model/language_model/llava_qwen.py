#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch.nn.functional as F
import torch
import wandb

import torch.nn as nn

import numpy as np

from typing import List, Optional, Tuple, Union, Any, Mapping

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.models.qwen2.modeling_qwen2 import QWEN2_INPUTS_DOCSTRING, _CONFIG_FOR_DOC

from vggt.train_utils.normalization import normalize_camera_extrinsics_and_points_batch

import os
from cambrianp.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # config.num_hidden_layers = 1 # NOTE: for debug only!!!
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)        
        # Initialize weights and apply final processing
        self.post_init()
        
        self.load_rec_model = getattr(self.config, "load_rec_model", False)
            
    def get_model(self):
        return self.model


    @staticmethod
    def move_special_tensors_to_cuda(obj, keys=("img_mask", "ray_mask",), device="cuda:0"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in keys and isinstance(v, torch.Tensor):
                    obj[k] = v.to(device).to(torch.bfloat16)
                else:
                    obj[k] = LlavaQwenForCausalLM.move_special_tensors_to_cuda(v, keys, device)
            return obj
        elif isinstance(obj, list):
            return [LlavaQwenForCausalLM.move_special_tensors_to_cuda(item, keys, device) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.to(device).to(torch.bfloat16)
        elif isinstance(obj, np.ndarray):
            return torch.from_numpy(obj).to(device).to(torch.bfloat16)
        else:
            return obj
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        rec_views: Optional[Any] = None,
        has_rec_views_mask: Optional[List[bool]] = None,
        bp_vqa: Optional[bool] = None,
        bp_rec: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:

            (input_ids, position_ids,
             attention_mask, past_key_values,
             inputs_embeds, labels,
             info_dict_list) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values,
                labels, images, modalities, image_sizes, bp_rec=bp_rec,
                bp_vqa=bp_vqa,
            )
             
        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]

            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            torch.cuda.empty_cache()

            outputs = self.llm_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=(self.load_rec_model if self.load_rec_model else output_hidden_states),
                return_dict=return_dict,
                custom_llm_ce_loss=getattr(self.config, "custom_llm_ce_loss", False),
            )

            if self.training:
                vqa_loss_val = outputs.loss.item()
                vqa_count = 1.0
                
                vqa_loss_tensor = torch.tensor([vqa_loss_val], device=self.model.device)
                vqa_count_tensor = torch.tensor([vqa_count], device=self.model.device)
                
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(vqa_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(vqa_count_tensor, op=torch.distributed.ReduceOp.SUM)
                
                total_vqa_loss = vqa_loss_tensor.item()
                total_vqa_count = vqa_count_tensor.item()
                
                if not hasattr(self, '_accumulated_vqa_loss'):
                    self._accumulated_vqa_loss = 0.0
                    self._accumulated_vqa_count = 0.0
                
                # Accumulate
                self._accumulated_vqa_loss += total_vqa_loss
                self._accumulated_vqa_count += total_vqa_count
                
                # Compute average for this step (trainer will read this)
                if self._accumulated_vqa_count > 0:
                    self._last_vqa_loss = self._accumulated_vqa_loss / self._accumulated_vqa_count
                else:
                    self._last_vqa_loss = None
                 
            # subsample the rec_views
            if bp_rec is not None:
                bp_rec = torch.tensor(bp_rec, device=self.model.device)

            # To remove the rec_views and outputs if no bp_rec
            if self.training and self.load_rec_model and bp_rec.sum() < bp_rec.shape[0]:
                # The subsample-based implementation below blocks training; keep it commented out.
                # rec_views, outputs, info_dict_list = self.subsample_by_bp_rec(outputs, info_dict_list, rec_views, bp_rec)

                # Use torch.where to properly stop gradients for samples where bp_rec=False
                # The original in-place assignment (tensor[mask] = tensor[mask].detach()) doesn't
                # actually break the gradient graph - it only copies values.
                new_hidden_states = []
                for hidden_state in outputs.hidden_states:
                    # Expand bp_rec to match hidden_state shape: (B,) -> (B, 1, 1, ...)
                    bp_rec_expanded = bp_rec.view(-1, *([1] * (hidden_state.dim() - 1)))
                    # torch.where: keeps gradients where bp_rec=True, uses detached where bp_rec=False
                    new_hs = torch.where(bp_rec_expanded, hidden_state, hidden_state.detach())
                    new_hidden_states.append(new_hs)
                outputs.hidden_states = tuple(new_hidden_states)
            
            if self.training and self.load_rec_model and rec_views is not None and sum(has_rec_views_mask) > 0:
                rec_views, rec_outputs, rec_info_dict_list = _process_batch_vggt(
                    rec_views, outputs, info_dict_list, has_rec_views_mask
                )

                rec_loss_dict, rec_predictions = self.model.rec_head(rec_outputs, info_dict_list=rec_info_dict_list, batch=rec_views)
            
                rec_loss_weight = getattr(self.config, "rec_loss_weight", 1.0)
                
                # Store reconstruction metrics for trainer logging
                # This will be picked up by the LLaVATrainer for tqdm display
                self._last_rec_loss_dict = {
                    "rec_all": round(rec_loss_dict["objective"].item(), 4),
                    **{f"rec_{key}": round(self._to_scalar(value), 4) for key, value in rec_loss_dict.items() if key != "objective"}
                }

                outputs.loss += rec_loss_weight * rec_loss_dict["objective"]
                
                if (getattr(self.config, "enable_rec_pose_viz", False)
                    and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)):
                    self._rec_pose_viz_counter = getattr(self, '_rec_pose_viz_counter', 0) + 1
                    if self._rec_pose_viz_counter % getattr(self.config, "rec_pose_viz_frequency", 100) == 0:
                        self._visualize_rec_head_poses(rec_predictions, rec_views, getattr(self, '_wandb_step', 0))

            elif self.training and self.load_rec_model:
                dummy_rec_loss = sum(p.sum() for p in self.model.rec_head.parameters()) * 0
                outputs.loss = outputs.loss + dummy_rec_loss
            
            if not self.training and self.model.load_rec_model and getattr(self.config, "use_camera_tokens", False):
                # Dummy rec_views for the generating phase.
                batch_size = 1
                rec_views = {
                    'images': torch.randn(batch_size, 32, 3, 192, 192).to(dtype=torch.bfloat16, device=self.model.device),
                }
                try:
                    #  Only need to forward the rec_head once for generation
                    if hasattr(self.model, "rec_head") and getattr(self, "rec_head_preds", None) is None:
                        predictions = self.model.rec_head(outputs, info_dict_list=self.info_dict_list, batch=rec_views)
                        self.rec_head_preds = predictions       
                except Exception as e:
                    print(f"Error in generating rec_head predictions: {e}")
                    
            return outputs
    
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def llm_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        custom_llm_ce_loss: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            if custom_llm_ce_loss:
                loss = nn.functional.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), reduction="none")
                loss = (loss.view(shift_logits.size(0), shift_logits.size(1)).sum(-1) / (~shift_labels.eq(-100)).sum(-1)).mean()
            else:
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        rec_views: Optional[Any] = None,
        bp_vqa: Optional[bool] = None,
        bp_rec: Optional[bool] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _,
             info_dict_list) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, modalities,
                image_sizes=image_sizes, bp_rec=bp_rec, bp_vqa=bp_vqa,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            info_dict_list = None
        
        # Ensure output_hidden_states is True if we need rec_head processing
        
        # kwargs['output_hidden_states'] = True
        # kwargs['return_dict_in_generate'] = True
        self.info_dict_list = info_dict_list
        
        # Generate the text output
        generation_output = super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        
        return generation_output, getattr(self, "rec_head_preds", None)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
            
        return inputs
    
    def _visualize_rec_head_poses(self, predictions, rec_views, step):
        """Visualize rec_head camera pose predictions."""
        
        from cambrianp.camera_trajectory_utils import RecHeadPoseVisualizer
        visualizer = RecHeadPoseVisualizer(self.config)
        visualizer.visualize_rec_head_poses(predictions, rec_views, step)

    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.float().cpu().numpy()
        return np.asarray(tensor)
    
    def _to_scalar(self, x):
        if isinstance(x, torch.Tensor):
            return x.item()
        return float(x)
        
        
def _process_batch_vggt(batch: Mapping, outputs, info_dict_list, has_rec_views_mask=None):
    """
    Process rec_views batch and subset outputs/info_dict_list to align with rec_views.
    
    Args:
        batch: rec_views dict with extrinsics, cam_points, world_points, depths, etc.
        outputs: model outputs containing hidden_states
        info_dict_list: list of info dicts, one per batch sample
        has_rec_views_mask: boolean mask indicating which samples have rec_views (works with FSDP/DDP)
    
    Returns:
        batch: processed rec_views batch
        rec_outputs: subset of outputs aligned with rec_views
        rec_info_dict_list: subset of info_dict_list aligned with rec_views
    """
    # Handle mixed batch where only some samples have rec_views (e.g., video files don't)
    rec_outputs = outputs
    rec_info_dict_list = info_dict_list
    if has_rec_views_mask is not None:
        # Convert to tensor for boolean indexing
        mask = torch.tensor(has_rec_views_mask, dtype=torch.bool)
        # Only subset if not all samples have rec_views
        if not mask.all():
            rec_hidden_states = tuple(
                hs[mask] for hs in outputs.hidden_states
            )
            rec_outputs = type(outputs)(
                loss=outputs.loss,
                logits=outputs.logits[mask] if outputs.logits is not None else None,
                past_key_values=None,
                hidden_states=rec_hidden_states,
                attentions=None,
            )
            rec_info_dict_list = [info_dict_list[i] for i, has_rv in enumerate(has_rec_views_mask) if has_rv]
    
    device = batch["extrinsics"].device
    
    # move to cpu
    batch["extrinsics"] = batch["extrinsics"].cpu()
    batch["cam_points"] = batch["cam_points"].cpu()
    batch["world_points"] = batch["world_points"].cpu()
    batch["depths"] = batch["depths"].cpu()
    batch["point_masks"] = batch["point_masks"].cpu()
    batch["scale_by_points"] = batch["scale_by_points"].cpu()
    
    sbp = batch.get("scale_by_points", False)
    try:
        if isinstance(sbp, bool):
            sbp_flag = sbp
        elif torch.is_tensor(sbp):
            sbp_flag = bool(sbp.detach().cpu().flatten().any().item())
        elif isinstance(sbp, np.ndarray):
            sbp_flag = bool(np.asarray(sbp).flatten().any())
        else:
            sbp_flag = bool(sbp)
    except Exception:
        sbp_flag = bool(sbp)
       
    # Normalize camera extrinsics and points. The function returns new tensors.
    normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            scale_by_points=sbp_flag,
            point_masks=batch["point_masks"],
        )
    # Replace the original values in the batch with the normalized ones.
    batch["extrinsics"] = normalized_extrinsics
    batch["cam_points"] = normalized_cam_points
    batch["world_points"] = normalized_world_points
    batch["depths"] = normalized_depths

    # move to gpu
    batch["extrinsics"] = batch["extrinsics"].to(device)
    batch["cam_points"] = batch["cam_points"].to(device)
    batch["world_points"] = batch["world_points"].to(device)
    batch["depths"] = batch["depths"].to(device)
    batch["point_masks"] = batch["point_masks"].to(device)
    
    # Preserve is_metric_scale for loss computation (default True for rec data)
    if "is_metric_scale" in batch:
        batch["is_metric_scale"] = batch["is_metric_scale"].to(device)
    else:
        # Default: assume metric scale for non-MapAnything data
        B = batch["extrinsics"].shape[0]
        batch["is_metric_scale"] = torch.ones(B, dtype=torch.bool, device=device)
    
    return batch, rec_outputs, rec_info_dict_list

AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
