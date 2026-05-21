import copy
import json
import logging
import math
import os
import re
import sys
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import av
import numpy as np
import PIL
import torch
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm

from transformers import AutoConfig
from PIL import Image


from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
from pathlib import Path
# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

# cambrian_p depends on the cambrian-p model code and the vggt 3D-vision repo;
# set CAMBRIAN_P_PATH / CAMBRIAN_P_VGGT_PATH to point at their clones.
# TODO: remove this block once cambrian-p ships with vggt as an installable dep.
for _env_var in ("CAMBRIAN_P_PATH", "CAMBRIAN_P_VGGT_PATH"):
    _path = os.environ.get(_env_var)
    if _path and os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from cambrianp.model.language_model.llava_qwen import LlavaQwenConfig

# Import LLaVA modules
from cambrianp.constants import (
    CAM_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from cambrianp.conversation import SeparatorStyle, conv_templates
from cambrianp.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from cambrianp.model.builder import load_pretrained_model


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


SAFE_DECORD_CHUNK_SIZE = 16
SUSPICIOUS_FRAME_COUNT_RATIO = 1.75
DEFAULT_DECORD_NUM_THREADS = 0
SAFE_DECORD_NUM_THREADS = 1


@register_model("cambrian_p")
class CambrianP(lmms):
    """
    Cambrian-P model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "vicuna_v1",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        use_camera_tokens: Optional[bool] = False,  # init camera tokens
        camera_tokens_mode: Optional[str] = "camera_tokens",
        camera_tokens_place: Optional[str] = "prepend_to_frame",
        query_mode: Optional[str] = "query_after_question",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend
        self.use_camera_tokens = use_camera_tokens
        self.camera_tokens_mode = camera_tokens_mode
        self.query_mode = query_mode
        self.camera_tokens_place = camera_tokens_place
        self._force_safe_decord_paths = set()
        self._safe_decord_path_decisions = {}
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode

        # Add camera tokens initialization parameter to config
        overwrite_config["use_camera_tokens"] = self.use_camera_tokens
        overwrite_config["camera_tokens_mode"] = self.camera_tokens_mode
        overwrite_config["query_mode"] = self.query_mode
        overwrite_config["camera_tokens_place"] = self.camera_tokens_place


        if '7b' not in pretrained.lower():
            overwrite_config["tie_word_embeddings"] = True
        
        overwrite_config["camera_token_indices"] = [0]
        
        cfg_pretrained = LlavaQwenConfig.from_pretrained(self.pretrained)
        
        if "cambrian" in self.pretrained.lower():
            # start to cast the config to LlavaQwenConfig
            from cambrianp.utils import cast_cambrian_config_to_llava_ov_style
            cfg_pretrained = cast_cambrian_config_to_llava_ov_style(cfg_pretrained, self.pretrained)


        if cfg_pretrained.architectures[0] == "LlavaLlamaForCausalLM":  # Ugly code, only used in  vicuna that needs ROPE
            if "224" in cfg_pretrained.mm_vision_tower:
                least_token_number = self.max_frames_num * (16 // self.mm_spatial_pool_stride) ** 2 + 1000
            else:
                least_token_number = self.max_frames_num * (24 // self.mm_spatial_pool_stride) ** 2 + 1000

            scaling_factor = math.ceil(least_token_number / 4096)
            if scaling_factor >= 2:
                overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

        llava_model_args["overwrite_config"] = overwrite_config
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)

        self._config = self._model.config
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1


    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            if visual is None or visual == []:
                visual = None
                task_type = "text"
                image_tensor = None
            else:
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:
                    self._config.image_aspect_ratio = "pad"
                    eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                if "task_type" in self.metadata and self.metadata["task_type"] == "video" and "sample_frames" in self.metadata:
                    assert type(visual) == list, "sample_frames must be specified for video task"
                    sample_indices = np.linspace(0, len(visual) - 1, self.metadata["sample_frames"], dtype=int)
                    visual = [visual[i] for i in sample_indices]
                    assert len(visual) == self.metadata["sample_frames"]

                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                    task_type = "video"

                elif type(visual[0]) == PIL.Image.Image:
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                    task_type = "image"

                elif type(visual[0]) == str:
                    image_tensor = []
                    try:
                        frames = self._load_video_frames(visual, self.max_frames_num)
                        frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                        image_tensor.append(frames)
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        image_tensor = None

                    task_type = "video"

            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in contexts:
                placeholder_count = len(visual) if isinstance(visual, list) else 1
                if task_type == "video":
                    placeholder_count = len(frames) if self.token_strategy == "multiple" else 1
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts
            else:
                prompts_input = contexts

            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            conv.messages[-1][1] = continuation
            full_prompt = conv.get_prompt()
            full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            labels = full_input_ids.clone()
            labels[0, : input_ids.shape[1]] = -100

            kwargs = {}
            if task_type == "image":
                kwargs["image_sizes"] = [[v.size[0], v.size[1]] for v in visual] if isinstance(visual, list) else [[visual.size[0], visual.size[1]]]
            elif task_type == "video":
                kwargs["modalities"] = ["video"]
                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            with torch.inference_mode():
                outputs = self.model(input_ids=full_input_ids, labels=labels, images=image_tensor, use_cache=True, **kwargs)

            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = full_input_ids[:, input_ids.shape[1] :]
            greedy_tokens = greedy_tokens[:, input_ids.shape[1] : full_input_ids.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()

            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _resolve_video_path(self, video_path):
        return video_path if isinstance(video_path, str) else video_path[0]

    def _estimate_video_frame_count(self, video_path):
        container = None
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            if stream.average_rate is None:
                return None

            fps = float(stream.average_rate)
            if fps <= 0:
                return None

            duration_seconds = None
            if stream.duration is not None and stream.time_base is not None:
                duration_seconds = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration_seconds = float(container.duration / av.time_base)

            if duration_seconds is None or duration_seconds <= 0:
                return None

            return duration_seconds * fps
        except Exception as exc:
            eval_logger.debug(f"Failed to estimate frame count for {video_path}: {exc}")
            return None
        finally:
            if container is not None:
                container.close()

    def _should_use_safe_decord_path(self, video_path, total_frame_num):
        if video_path in self._force_safe_decord_paths:
            self._safe_decord_path_decisions[video_path] = True
            return True

        cached_decision = self._safe_decord_path_decisions.get(video_path)
        if cached_decision is not None:
            return cached_decision

        estimated_frame_count = self._estimate_video_frame_count(video_path)
        if estimated_frame_count is None or estimated_frame_count <= 0 or total_frame_num <= 0:
            self._safe_decord_path_decisions[video_path] = False
            return False

        mismatch_ratio = max(total_frame_num / estimated_frame_count, estimated_frame_count / total_frame_num)
        if mismatch_ratio >= SUSPICIOUS_FRAME_COUNT_RATIO:
            eval_logger.warning(
                f"Suspicious frame count metadata for {video_path}: decord reports {total_frame_num} "
                f"frames while stream metadata suggests about {estimated_frame_count:.1f}. "
                "Switching to safe Decord loading."
            )
            self._force_safe_decord_paths.add(video_path)
            self._safe_decord_path_decisions[video_path] = True
            return True

        self._safe_decord_path_decisions[video_path] = False
        return False

    def _load_video_chunked(self, video_path, frame_idx, chunk_size=SAFE_DECORD_CHUNK_SIZE):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=SAFE_DECORD_NUM_THREADS)
        frame_batches = []
        chunk_size = max(1, min(chunk_size, len(frame_idx)))
        for i in range(0, len(frame_idx), chunk_size):
            chunk_indices = frame_idx[i : i + chunk_size]
            batch = vr.get_batch(chunk_indices).asnumpy()
            vr.seek(0)
            frame_batches.append(batch)

        if frame_batches:
            return np.concatenate(frame_batches, axis=0)
        return np.array([])

    def load_video(self, video_path, max_frames_num):
        video_path = self._resolve_video_path(video_path)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=DEFAULT_DECORD_NUM_THREADS)
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()

        if self._should_use_safe_decord_path(video_path, total_frame_num):
            try:
                return self._load_video_chunked(video_path, frame_idx)
            except Exception as exc:
                eval_logger.warning(f"Safe Decord loading failed for {video_path}: {exc}. Falling back to PyAV.")
                return read_video_pyav(video_path, num_frm=max_frames_num)

        try:
            return vr.get_batch(frame_idx).asnumpy()
        except Exception as exc:
            self._force_safe_decord_paths.add(video_path)
            self._safe_decord_path_decisions[video_path] = True
            eval_logger.warning(
                f"Decord batch loading failed for {video_path}: {exc}. Retrying with chunked Decord loading."
            )
            try:
                return self._load_video_chunked(video_path, frame_idx)
            except Exception as chunk_exc:
                eval_logger.warning(
                    f"Chunked Decord loading failed for {video_path}: {chunk_exc}. Falling back to PyAV."
                )
                return read_video_pyav(video_path, num_frm=max_frames_num)

    def _load_video_frames(self, video_path, max_frames_num):
        resolved_video_path = self._resolve_video_path(video_path)
        if self.video_decode_backend == "decord":
            return self.load_video(resolved_video_path, max_frames_num)
        elif self.video_decode_backend == "pyav":
            return read_video_pyav(resolved_video_path, num_frm=max_frames_num)
        else:
            raise ValueError(f"Unsupported video_decode_backend: {self.video_decode_backend}")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []

            video_path_for_features = None
            image_size = None
            for visual, context in zip(batched_visuals, batched_contexts):
                if visual is None or visual == []:  # for text-only tasks.
                    visual = None
                    task_type = "text"
                    placeholder_count = 0
                    image_tensor = None
                else:
                    if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                        self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                        eval_logger.info(f"In Multi-Image setting, image aspect ratio: {self._config.image_aspect_ratio}")

                    if "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:  # overwrite logic for video task with multiple static image frames
                        assert type(visual) == list, "sample_frames must be specified for video task"
                        sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                        visual = [visual[i] for i in sample_indices]
                        assert len(visual) == metadata["sample_frames"]

                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "video"
                        placeholder_count = 1

                    # elif type(visual[0]) == PIL.Image.Image:  # For image, multi-image tasks
                    elif isinstance(visual[0], PIL.Image.Image): # ! NOTE@sy: '==' is too hard
                        image_size = [_.size for _ in visual]
                        image_tensor = process_images(visual, self._image_processor, self._config)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                        task_type = "image"
                        placeholder_count = len(visual) if isinstance(visual, list) else 1

                    elif type(visual[0]) == str:  # For video task
                        video_path_for_features = visual[0]  # video path names
                        image_tensor = []
                        try:
                            frames = self._load_video_frames(visual, self.max_frames_num)

                            image_size = list(frames.shape[1:3])
                            if "cambrian" in self.pretrained.lower():
                                frames = [expand2square(Image.fromarray(frames[_], mode="RGB"), tuple(int(x*255) for x in self._image_processor.image_mean)) for _ in range(frames.shape[0])]

                            frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                            image_tensor.append(frames)
                        except Exception as e:
                            eval_logger.error(f"Error {e} in loading video")
                            image_tensor = None

                        task_type = "video"
                        placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                    """
                    # if task_type == "image": # indeed in multi-image case, not the video in frames.
                    #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    # elif task_type == "video":
                    # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                # import ipdb; ipdb.set_trace(context=20)
                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    if self.camera_tokens_mode == 'camera_tokens' and self.camera_tokens_place == 'between_qa':
                        # Add <cam> at the end of the question before where answer would be
                        self.cam_token_index = self.tokenizer.convert_tokens_to_ids(CAM_TOKEN)
                        self.model.cam_token_index = self.cam_token_index # also set is in llava_arch
                        question = question.rstrip() + '<cam>'
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            text_outputs = self._run_inference_step(
                input_ids, attention_masks, image_tensor, image_size,
                task_type, gen_kwargs, video_path_for_features,
            )
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def _run_inference_step(self, input_ids, attention_masks, image_tensor, image_size,
                            task_type, gen_kwargs, video_path_for_features):
        # preconfigure gen_kwargs with defaults
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        if task_type == "image":
            # gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            ...
        elif task_type == "video":
            conv = conv_templates[self.conv_template].copy()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            gen_kwargs["modalities"] = ["video"]
            gen_kwargs["stopping_criteria"] = [stopping_criteria]
            self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
            self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

        # These steps are not in LLaVA's original code, but are necessary for generation to work
        # TODO: attention to this major generation step...
        if "image_aspect_ratio" in gen_kwargs.keys():
            gen_kwargs.pop("image_aspect_ratio")

        try:
            with torch.inference_mode():
                cont, _ = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, image_sizes=image_size, **gen_kwargs)

            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        except Exception as e:
            raise e

        text_outputs = [response.strip() for response in text_outputs]
        # print(self.tokenizer.batch_decode(input_ids % self.tokenizer.vocab_size), text_outputs)
        return text_outputs


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
