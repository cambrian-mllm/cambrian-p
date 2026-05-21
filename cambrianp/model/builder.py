#    Copyright 2023 Haotian Liu
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


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from cambrianp.model import *
from cambrianp.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrianp.utils import rank0_print, load_model_state_dict, verify_model_weights, load_camera_tokens_into_model
import torch.nn as nn


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", torch_dtype="float16",attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        import pdb;pdb.set_trace()

    if customized_config is not None:
        kwargs["config"] = customized_config

    if "multimodal" in kwargs:
        if kwargs["multimodal"] is True:
            is_multimodal = True
            kwargs.pop("multimodal")
    else:
        is_multimodal = False

    if "llava" in model_name.lower() or is_multimodal:
        # Load LLaVA model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if "lora" in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            rank0_print("Loading LLaVA from base model...")
            if "mixtral" in model_name.lower():
                from cambrianp.model.language_model.llava_mixtral import LlavaMixtralConfig

                lora_cfg_pretrained = LlavaMixtralConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "mistral" in model_name.lower():
                from cambrianp.model.language_model.llava_mistral import LlavaMistralConfig

                lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "gemma" in model_name.lower():
                from cambrianp.model.language_model.llava_gemma import LlavaGemmaConfig

                lora_cfg_pretrained = LlavaGemmaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaGemmaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            else:
                from cambrianp.model.language_model.llava_llama import LlavaConfig

                lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)

            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            rank0_print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
            non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            rank0_print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            rank0_print("Merging LoRA weights...")
            model = model.merge_and_unload()
            rank0_print("Model is loaded...")
        elif model_base is not None:  # this may be mm projector only, loading projector with preset language mdoel
            rank0_print(f"Loading LLaVA from base model {model_base}...")
            if "mixtral" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "gemma" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaGemmaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif (
                "wizardlm-2" in model_name.lower()
                and "vicuna" in model_name.lower()
                or "llama" in model_name.lower()
                or "yi" in model_name.lower()
                or "nous-hermes" in model_name.lower()
                or "llava-v1.6-34b" in model_name.lower()
                or "llava-v1.5" in model_name.lower()
            ):
                from cambrianp.model.language_model.llava_llama import LlavaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaConfig.from_pretrained(model_path)
                    if "v1.5" in model_name.lower():
                        llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                else:
                    llava_cfg = customized_config

                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                llava_cfg = LlavaConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=llava_cfg, **kwargs)
            else:
                raise ValueError(f"Model {model_name} not supported")

            mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            rank0_print(f"Loaded LLaVA model: {model_path}")
            if "mixtral" in model_name.lower():
                from cambrianp.model.language_model.llava_mixtral import LlavaMixtralConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaMixtralConfig.from_pretrained(model_path)
                else:
                    llava_cfg = customized_config

                if overwrite_config is not None:
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(llava_cfg, k, v)

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMixtralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)

            elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)
            elif (
                "wizardlm-2" in model_name.lower()
                and "vicuna" in model_name.lower()
                or "llama" in model_name.lower()
                or "yi" in model_name.lower()
                or "nous-hermes" in model_name.lower()
                or "llava-v1.6-34b" in model_name.lower()
                or "llava-v1.5" in model_name.lower()
            ):
                from cambrianp.model.language_model.llava_llama import LlavaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaConfig.from_pretrained(model_path)
                    if "v1.5" in model_name.lower():
                        llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                else:
                    llava_cfg = customized_config

                if overwrite_config is not None:
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(llava_cfg, k, v)

                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)

            elif "qwen" in model_name.lower() or "quyen" in model_name.lower() or "cambrian" in model_name.lower():
                # CHECK IF THE MODEL IS A FINE-TUNED LLAVA-VIDEO MODEL
                pretrained_model_name_or_path = model_path.strip("/").split("/")[-1]
                if pretrained_model_name_or_path.startswith("ft-llava-video"):
                    tokenizer = AutoTokenizer.from_pretrained("lmms-lab/LLaVA-Video-7B-Qwen2")
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                if "moe" in model_name.lower() or "A14B" in model_name.lower():
                    from cambrianp.model.language_model.llava_qwen_moe import LlavaQwenMoeConfig
                    if overwrite_config is not None:
                        llava_cfg = LlavaQwenMoeConfig.from_pretrained(model_path)
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                        model = LlavaQwenMoeForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
                    else:
                        model = LlavaQwenMoeForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, **kwargs)

                else:
                    from cambrianp.model.language_model.llava_qwen import LlavaQwenConfig
                    llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
                    
                    if "cambrian" in model_path.lower():
                        # start to cast the config to LlavaQwenConfig
                        from cambrianp.utils import cast_cambrian_config_to_llava_ov_style
                        llava_cfg = cast_cambrian_config_to_llava_ov_style(llava_cfg, model_path)
                    
                    if overwrite_config is not None:
                        rank0_print(f"Overwriting config with {overwrite_config}")
                        for k, v in overwrite_config.items():
                            setattr(llava_cfg, k, v)
                    
                    is_cambrian = 'cambrian' in model_path.lower()
                    new_state_dict = load_model_state_dict(model_path, remap_cambrian=is_cambrian)
                    
                    model = LlavaQwenForCausalLM.from_pretrained(
                        model_path, 
                        low_cpu_mem_usage=True, 
                        attn_implementation=attn_implementation, 
                        config=llava_cfg, 
                        **kwargs
                    )
                    
                    if new_state_dict is not None:
                        model.load_state_dict(new_state_dict, strict=False)
                        load_camera_tokens_into_model(model, new_state_dict)
                        verify_model_weights(model, new_state_dict)
                        
                        del new_state_dict
                        torch.cuda.empty_cache()
                    
            elif "gemma" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaGemmaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                from cambrianp.model.language_model.llava_qwen import LlavaQwenConfig
                llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
                
                if overwrite_config is not None:
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(llava_cfg, k, v)
                
                is_cambrian = 'cambrian' in model_path.lower()
                new_state_dict = load_model_state_dict(model_path, remap_cambrian=is_cambrian)
                
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_path, 
                    low_cpu_mem_usage=True, 
                    attn_implementation=attn_implementation, 
                    config=llava_cfg, 
                    **kwargs
                )
                
                if new_state_dict is not None:
                    model.load_state_dict(new_state_dict, strict=False)
                    load_camera_tokens_into_model(model, new_state_dict)
                    
                    verify_model_weights(model, new_state_dict)
                    
                    del new_state_dict
                    torch.cuda.empty_cache()

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    rank0_print(f"Model Class: {model.__class__.__name__}")
    image_processor = None

    if "llava" in model_name.lower() or is_multimodal:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            vision_tower.to(device="cuda", dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
