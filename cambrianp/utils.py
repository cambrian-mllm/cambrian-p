import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np
import imageio
import torch.nn as nn
import cv2
import json
from safetensors.torch import load_file
from safetensors import safe_open
import torch
import copy

import requests

from cambrianp.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."

handler = None

import torch.distributed as dist

try:
    import av
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")


def process_video_with_decord(video_file, data_args):
    try:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    except Exception as e:
        raise RuntimeError(f"Error loading video {video_file}: {e}")
    
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i/avg_fps for i in frame_idx]

    
    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample


def process_video_with_decord_byframe(
    video_file, 
    data_args, 
    start_frame: int, 
    end_frame: int, 
    current_observation_frame: int = None
):
    """
    Load video frames by frame range (from Cambrian-S).
    
    Args:
        video_file: Path to video file
        data_args: Data arguments with video_max_frames/frames_upbound, video_fps, force_sample
        start_frame: Start frame index
        end_frame: End frame index  
        current_observation_frame: Optional frame to append at end (for NFP)
    
    Returns:
        video: numpy array (N, H, W, C)
        video_time: Duration in seconds
        frame_time: Comma-separated frame times
        num_frames_to_sample: Number of frames
    """
    try:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        selected_frame = min(total_frame_num - 1, end_frame)
        
        assert start_frame < selected_frame, f"start_frame {start_frame} must be less than selected_frame {selected_frame}"
        
        video_fps = getattr(data_args, 'video_fps', 1)
        avg_fps = round(vr.get_avg_fps() / video_fps) if video_fps > 0 else 1
        frame_idx = [i for i in range(start_frame, selected_frame, max(1, avg_fps))]

        video_time = (selected_frame - start_frame) / avg_fps

        # Get max frames from either video_max_frames or frames_upbound
        video_max_frames = getattr(data_args, 'video_max_frames', 0) or getattr(data_args, 'frames_upbound', 0)
        force_sample = getattr(data_args, 'video_force_sample', False) or getattr(data_args, 'force_sample', False)
        
        if video_max_frames > 0:
            target_frames = video_max_frames
            if current_observation_frame is not None:
                target_frames -= 1
            if len(frame_idx) > target_frames or force_sample:
                uniform_sampled_frames = np.linspace(start_frame, selected_frame, target_frames, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()

        frame_time = [(i - start_frame) / avg_fps for i in frame_idx]
        
        if current_observation_frame is not None:
            frame_idx.append(current_observation_frame)
            frame_time.append((current_observation_frame - start_frame) / avg_fps)
            
        frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])

        video = vr.get_batch(frame_idx).asnumpy()
        num_frames_to_sample = len(frame_idx)
        vr.seek(0)
    except Exception as e:
        raise RuntimeError(f"Video processing error for {video_file}: {e}")
    
    return video, video_time, frame_time_str, num_frames_to_sample


def process_video_with_decord_bytime(
    video_file, 
    data_args, 
    start_time: float, 
    end_time: float
):
    """
    Load video frames by time range (from Cambrian-S).
    
    Args:
        video_file: Path to video file
        data_args: Data arguments
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns:
        video: numpy array (N, H, W, C)
        video_time: Duration in seconds
        frame_time: Comma-separated frame times
        num_frames_to_sample: Number of frames
    """
    try:
        video_time = end_time - start_time
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        fps = vr.get_avg_fps()

        video_fps = getattr(data_args, 'video_fps', 1)
        avg_fps = round(fps / video_fps) if video_fps > 0 else 1
        
        start_frame = int(start_time * fps)
        end_frame = min(int(end_time * fps), total_frame_num - 1)
        
        frame_idx = [i for i in range(start_frame, end_frame, max(1, avg_fps))]
        frame_time = [(i - start_frame) / avg_fps for i in frame_idx]

        video_max_frames = getattr(data_args, 'video_max_frames', 0) or getattr(data_args, 'frames_upbound', 0)
        force_sample = getattr(data_args, 'video_force_sample', False) or getattr(data_args, 'force_sample', False)
        
        if video_max_frames > 0:
            if len(frame_idx) > video_max_frames or force_sample:
                uniform_sampled_frames = np.linspace(start_frame, end_frame, video_max_frames, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frame_time = [(i - start_frame) / avg_fps for i in frame_idx]

        frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
        video = vr.get_batch(frame_idx).asnumpy()
        num_frames_to_sample = len(frame_idx)
        vr.seek(0)
    except Exception as e:
        raise RuntimeError(f"Video processing error for {video_file}: {e}")
    
    return video, video_time, frame_time_str, num_frames_to_sample


def process_gif_with_imageio(video_file, data_args):
    # ! NOTE: we treat gif as video
    try:
        gif = imageio.get_reader(video_file)
    except Exception:
        # Fallback to PIL for problematic GIFs
        from PIL import Image
        pil_gif = Image.open(video_file)
        frames = []
        try:
            while True:
                frames.append(np.array(pil_gif.convert('RGB')))
                pil_gif.seek(pil_gif.tell() + 1)
        except EOFError:
            pass
        video = np.stack(frames)
        num_frames = len(frames)
        return video, num_frames * 0.1, [i * 0.1 for i in range(num_frames)], num_frames
    
    num_frames = len(gif)
    video_time = num_frames * 0.1

    frame_idx = [i for i in range(0, num_frames, 1)]
    frame_time = [i * 0.1 for i in frame_idx]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            uniform_sampled_frames = np.linspace(0, num_frames - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i * 0.1 for i in frame_idx]

    video = []
    hw_set = set()
    min_h, min_w = 10000, 10000

    try:
        for index, frame in enumerate(gif):
            if index in frame_idx:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                frame = frame.astype(np.uint8)
                video.append(frame)
                hw_set.add(frame.shape)
                if frame.shape[0] < min_h:
                    min_h = frame.shape[0]
                if frame.shape[1] < min_w:
                    min_w = frame.shape[1]
    except ValueError as e:
        # Handle "No packer found from P to L" error
        if "packer" in str(e).lower():
            from PIL import Image
            pil_gif = Image.open(video_file)
            video = []
            frame_count = 0
            try:
                while True:
                    if frame_count in frame_idx:
                        video.append(np.array(pil_gif.convert('RGB')))
                    frame_count += 1
                    pil_gif.seek(pil_gif.tell() + 1)
            except EOFError:
                pass
        else:
            raise

    if len(hw_set) > 1:
        video = [frame[:min_h, :min_w] for frame in video]

    num_frames_to_sample = len(frame_idx)
    video = np.stack(video)
    return video, video_time, frame_time, num_frames_to_sample


def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    avg_fps = round(total_frame_num / video_time / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()


    frames = [video_frames[i] for i in frame_idx]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def cast_cambrian_config_to_llava_ov_style(config, model_path):
    # load the corresponding llava-ov config
    
    from cambrianp.model.language_model.llava_qwen import LlavaQwenConfig
    # according to the model_path to determine the llava-ov config
    if "0.5b" in model_path.lower():
        llava_ov_config = LlavaQwenConfig.from_pretrained("lmms-lab/llava-onevision-qwen2-0.5b-ov")
    elif "7b" in model_path.lower():
        llava_ov_config = LlavaQwenConfig.from_pretrained("lmms-lab/llava-onevision-qwen2-7b-ov")
    else:
        llava_ov_config = manual_cast_for_3B_and_1p5B_model(config)
        rank0_print(f"[cambrian-cast] Manually casted the config for 3B and 1.5B model for {model_path}")
    
    # check the overlapped keys, missing keys and unexpected keys between the config and the llava-ov config
    config_dict = config.to_dict()
    llava_ov_config_dict = llava_ov_config.to_dict()
    
    config_keys = set(config_dict.keys())
    llava_ov_keys = set(llava_ov_config_dict.keys())
    
    missing_keys = llava_ov_keys - config_keys
    unexpected_keys = config_keys - llava_ov_keys
    overlapped_keys = config_keys.intersection(llava_ov_keys)

    # Print a table for comparison
    rank0_print("\n" + "="*120)
    rank0_print(f"{'Key':<45} | {'Config Value':<35} | {'LLava-OV Value':<35}")
    rank0_print("-" * 120)
    for k in sorted(list(overlapped_keys)):
        val1 = config_dict[k]
        val2 = llava_ov_config_dict[k]
        if val1 != val2:
            v1 = str(val1)
            v2 = str(val2)
            # Truncate long values for display
            v1_disp = (v1[:32] + '..') if len(v1) > 34 else v1
            v2_disp = (v2[:32] + '..') if len(v2) > 34 else v2
            # Highlight differences in red
            rank0_print(f"{k:<45} | \033[91m{v1_disp:<35}\033[0m | \033[91m{v2_disp:<35}\033[0m")
    
    rank0_print("-" * 120)
    rank0_print("="*120 + "\n")

    # Update config based on llava_ov_config
    # 1. For overlapped keys (except those containing _name_or_path), use llava-OV's value
    for k in overlapped_keys:
        if "_name_or_path" != k:  # do not overwrite the _name_or_path
            old_val = getattr(config, k, None)
            new_val = llava_ov_config_dict[k]
            setattr(config, k, new_val)
            if old_val != new_val:
                rank0_print(f"\033[91m[cambrian-cast] Updated overlapped key: {k} = {new_val}\033[0m")
            else:
                rank0_print(f"[cambrian-cast] Updated overlapped key: {k} = {new_val}")

    # 2. For missing keys, add them to config from llava_ov_config
    for k in missing_keys:
        setattr(config, k, llava_ov_config_dict[k])
        rank0_print(f"[cambrian-cast] Added missing key: {k} = {llava_ov_config_dict[k]}")

    if hasattr(config, "mm_vision_tower_aux_list") and config.mm_vision_tower_aux_list:
        config.mm_vision_tower = config.mm_vision_tower_aux_list[0]
    config.model_type = "llava"
    if hasattr(config, "mm_img_tok_num"):
        saved_mm_img_tok_num = config.mm_img_tok_num
    elif hasattr(config, "miv_token_len"):
        saved_mm_img_tok_num = config.miv_token_len
    else:
        saved_mm_img_tok_num = None
    config.mm_newline_position = "grid"

    # 3. delete the keys that are not in the llava-ov config
    for k in unexpected_keys:
        delattr(config, k)
        rank0_print(f"[cambrian-cast] Deleted unexpected key: {k}")

    if saved_mm_img_tok_num is not None:
        config.mm_img_tok_num = saved_mm_img_tok_num
        rank0_print(f"[cambrian-cast] Preserved mm_img_tok_num = {saved_mm_img_tok_num}")
    
    return config


def load_model_state_dict(model_path, remap_cambrian=True) -> dict:
    if not os.path.isdir(model_path):
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(model_path)
    
    result_ckpt = {}
    
    # 1) Single-file case: model.safetensors
    single_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_path):
        rank0_print(f"[load_state_dict] Loading from single-file checkpoint: {single_path}")
        result_ckpt = load_file(single_path)
    
    # 2) Sharded case: model.safetensors.index.json
    elif os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        rank0_print(f"[load_state_dict] Loading from sharded checkpoint: {index_path}")
        
        with open(index_path, "r") as f:
            index = json.load(f)
            weight_map = index.get("weight_map", {})
        
        # Get unique shard files
        shard_files = sorted(set(weight_map.values()))
        
        # Load all shards
        for shard_name in shard_files:
            shard_path = os.path.join(model_path, shard_name)
            if not os.path.exists(shard_path):
                rank0_print(f"[load_state_dict] WARNING: Shard not found: {shard_path}")
                continue
            rank0_print(f"[load_state_dict] Loading shard: {shard_name}")
            shard_ckpt = load_file(shard_path)
            result_ckpt.update(shard_ckpt)
    
    else:
        rank0_print(f"[load_state_dict] No safetensors checkpoint found in {model_path}")
        return None
    
    rank0_print(f"[load_state_dict] Loaded {len(result_ckpt)} keys from checkpoint")
    
    def remap_key(k):
        # Cambrian: model.vision_tower_aux_list.0 -> model.vision_tower
        if remap_cambrian and k.startswith("model.vision_tower_aux_list.0."):
            k = k.replace("model.vision_tower_aux_list.0.", "model.vision_tower.", 1)
        
        # LayerScale: .ls1.weight/.ls2.weight -> .ls1.gamma/.ls2.gamma
        if '.ls1.weight' in k:
            k = k.replace('.ls1.weight', '.ls1.gamma')
        if '.ls2.weight' in k:
            k = k.replace('.ls2.weight', '.ls2.gamma')
        
        return k
    
    remapped_count = 0
    new_ckpt = {}
    for k, v in result_ckpt.items():
        new_key = remap_key(k)
        if new_key != k:
            remapped_count += 1
        new_ckpt[new_key] = v
    
    result_ckpt = new_ckpt
    
    if remapped_count > 0:
        rank0_print(f"[load_state_dict] Remapped {remapped_count} keys")
    
    return result_ckpt


def verify_model_weights(model, new_state_dict):
    rank0_print("Verifying model weights against new_state_dict...")
    model_state_dict = model.state_dict()
    mismatched = []
    for k, v in new_state_dict.items():
        if k not in model_state_dict:
            mismatched.append(f"Missing key: {k}")
        else:
            # Move to same device/dtype for comparison
            m_val = model_state_dict[k]
            v_val = v.to(m_val.device).to(m_val.dtype)
            if not torch.equal(m_val, v_val):
                mismatched.append(f"Value mismatch: {k}")
    
    if mismatched:
        red_text = lambda x: f"\033[91m{x}\033[0m"
        rank0_print(red_text(f"WARNING: {len(mismatched)} weights did not match!"))
        rank0_print(red_text(str(mismatched[:10])))
    else:
        green_text = lambda x: f"\033[92m{x}\033[0m"
        rank0_print(green_text("SUCCESS: All weights in new_state_dict match the loaded model."))


def load_camera_tokens_into_model(model, state_dict, dtype=torch.float16):
    """
    Load weights into `model` from an in-memory `state_dict`, with special handling for
    camera-token parameters that may not be properly registered for `load_state_dict`.

    Args:
        model: The (LLaVA) model to load weights into.
        state_dict: A PyTorch state dict mapping parameter names -> tensors. If a
            checkpoint dict is passed (containing a 'state_dict' key), it will be unwrapped.
        dtype: dtype to cast camera tokens to before assigning.
    """
    if state_dict is None:
        rank0_print("[cam-load] state_dict is None; skipping")
        return

    # Common checkpoint format: {"state_dict": {...}}
    tensors = state_dict.get("state_dict", state_dict) if isinstance(state_dict, dict) else state_dict

    if not isinstance(tensors, dict):
        raise TypeError(f"state_dict must be a dict-like mapping, got {type(tensors)}")

    rank0_print(f"[cam-load] Loading model weights from provided state_dict (keys={len(tensors)})")

    # Load the full state dict (best-effort), then patch camera tokens if needed.
    missing_keys, unexpected_keys = model.load_state_dict(tensors, strict=False)

    camera_token_keys = [
        "model.rec_head.camera_tokens",
        "model.camera_tokens",
        "model.rec_learnable_tokens",
    ]

    for key_name in camera_token_keys:
        if key_name not in tensors:
            continue

        cam = tensors[key_name].to(device=model.device, dtype=dtype)

        # Assign to the correct location
        if key_name == "model.rec_head.camera_tokens" and hasattr(model.model, "rec_head"):
            model.model.rec_head.camera_tokens = nn.Parameter(cam)
            rank0_print(
                f"[cam-load] ✓ Loaded {key_name}; "
                f"shape={cam.shape}, mean={cam.mean().item():.6f}, std={cam.std().item():.6f}"
            )
            if key_name in unexpected_keys:
                unexpected_keys.remove(key_name)
            break

        if key_name == "model.camera_tokens" and hasattr(model.model, "camera_tokens"):
            model.model.camera_tokens = nn.Parameter(cam)
            rank0_print(
                f"[cam-load] ✓ Loaded {key_name}; "
                f"shape={cam.shape}, mean={cam.mean().item():.6f}, std={cam.std().item():.6f}"
            )
            if key_name in unexpected_keys:
                unexpected_keys.remove(key_name)
            break

        if key_name == "model.rec_learnable_tokens" and hasattr(model.model, "camera_tokens"):
            model.model.camera_tokens = nn.Parameter(cam)
            rank0_print(
                f"[cam-load] ✓ Loaded {key_name} as camera_tokens; "
                f"shape={cam.shape}, mean={cam.mean().item():.6f}, std={cam.std().item():.6f}"
            )
            if key_name in unexpected_keys:
                unexpected_keys.remove(key_name)
            break

    # Log missing and unexpected keys
    if missing_keys:
        rank0_print(f"[cam-load] Missing keys ({len(missing_keys)}): {missing_keys[:10]}...")
    if unexpected_keys:
        rank0_print(f"[cam-load] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}...")

    if not missing_keys and not unexpected_keys:
        rank0_print("[cam-load] ✓ All keys loaded successfully")


def manual_cast_for_3B_and_1p5B_model(config):
    llava_ov_config = copy.deepcopy(config)
    llava_ov_config.do_sample = False
    llava_ov_config.torch_dtype = 'bfloat16'
    llava_ov_config.mm_vision_tower_lr = 2e-06
    llava_ov_config.image_aspect_ratio = 'anyres_max_9'
    llava_ov_config.max_window_layers = 24
    llava_ov_config.architectures = ['LlavaQwenForCausalLM']

    llava_ov_config.tokenizer_model_max_length = 32768
    
    llava_ov_config.model_type = "llava_qwen"
    llava_ov_config.transformers_version = '4.40.0.dev0'
    
    llava_ov_config.mm_tunable_parts = 'mm_vision_tower,mm_mlp_adapter,mm_language_model'
    llava_ov_config.image_token_index = 151646
    llava_ov_config.mm_newline_position = "grid"
    llava_ov_config.mm_spatial_pool_mode = 'bilinear'
    
    llava_ov_config.image_grid_pinpoints = [[384, 384], [384, 768], [384, 1152], [384, 1536], [384, 1920], [384, 2304], [768, 384], [768, 768], [768, 1152], [768, 1536], [768, 1920], [768, 2304], [1152, 384], [1152, 768], [1152, 1152], [1152, 1536], [1152, 1920], [1152, 2304], [1536, 384], [1536, 768], [1536, 1152], [1536, 1536], [1536, 1920], [1536, 2304], [1920, 384], [1920, 768], [1920, 1152], [1920, 1536], [1920, 1920], [1920, 2304], [2304, 384], [2304, 768], [2304, 1152], [2304, 1536], [2304, 1920], [2304, 2304]]
    
    llava_ov_config.use_pos_skipping = False
    llava_ov_config.image_split_resolution = None
    llava_ov_config.vision_tower_pretrained = None
    llava_ov_config.image_crop_resolution = None
    llava_ov_config.mm_vision_tower = 'google/siglip-so400m-patch14-384'
    llava_ov_config.pos_skipping_range = 4096
    llava_ov_config.mm_resampler_type = None
    llava_ov_config.mm_patch_merge_type = 'spatial_unpad'
    
    return llava_ov_config
