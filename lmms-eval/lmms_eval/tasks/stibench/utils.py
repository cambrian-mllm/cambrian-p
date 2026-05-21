import os
from collections import OrderedDict, defaultdict
from pathlib import Path
import yaml
from loguru import logger as eval_logger
import pandas as pd

import datasets

TASKS = [
    "Dimensional Measurement",
    "Spatial Relation",
    "3D Video Grounding",
    "Displacement & Path Length",
    "Speed & Acceleration",
    "Ego-Centric Orientation",
    "Trajectory Description",
    "Pose Estimation",
]

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "stibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def stibench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = os.path.join(cache_dir, "video", doc["Video"])
    assert os.path.exists(video_path), f"Video path: {video_path} does not exist."
    return [video_path]


def stibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # ! NOTE@sy: all questions are single choice questions
    question = doc["Question"]

    option_list = []
    for key, value in doc["Candidates"].items():
        option_list.append(f"{key}. {value}")
    option_text = "\n".join(option_list)

    # ! NOTE@sy: slightly different from VLMEvalkit's implementation which includes FPS info.

    return f"These are frames of a video.\n{question}\n{option_text}\nAnswer with the option's letter from the given choices directly."


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)

    return dataset


def stibench_process_results(doc, results):
    doc["prediction"] = results[0]
    doc["accuracy"] = exact_match(fuzzy_matching(doc["prediction"]), doc["Answer"])
    return {"stibench_score": doc}


def stibench_aggregate_results(results):
    results = pd.DataFrame(results)

    output = defaultdict(lambda: 0.0)
    for task_type, task_type_indexes in results.groupby("Task").groups.items():
        per_task_type = results.iloc[task_type_indexes]
        output[task_type] = per_task_type["accuracy"].mean().item()

    output["overall"] = results["accuracy"].mean().item()

    results = OrderedDict()
    results["overall"] = output["overall"] * 100.0
    for task_type in TASKS:
        results[task_type] = output[task_type] * 100.0

    tabulated_keys = ", ".join([_ for _ in results.keys()])
    tabulated_results = ", ".join([f"{_:.3f}" for _ in results.values()])
    eval_logger.info(f"Tabulated results: {tabulated_keys}")
    eval_logger.info(f"Tabulated results: {tabulated_results}")

    results["tabulated_keys"] = tabulated_keys
    results["tabulated_results"] = tabulated_results

    return results
