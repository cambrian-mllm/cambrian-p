import io
import os
import json
from loguru import logger as eval_logger
import numpy as np
from PIL import Image

import datasets
from collections import OrderedDict, defaultdict


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
cache_name = "cambrians_vsc"

def doc_to_visual(doc):
    video_path = doc["video_path"]
    video_path = os.path.join(base_cache_dir, cache_name, video_path)
    doc["video_path"] = video_path
    return [doc["video_path"]]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()

    question = "These are frames of a video.\n" + question.strip() + "\nPlease answer the question using a single word or phrase."
    return question


def process_docs_10mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "10mins")

def process_docs_30mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "30mins")

def process_docs_60mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "60mins")

def process_docs_120mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "120mins")

def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy

def to_float(x):
    try:
        return float(x)
    except BaseException as e:
        return 0.0

def process_results(doc, results):

    doc["prediction"] = float(results[0])
    doc["accuracy"] = mean_relative_accuracy(doc["prediction"], doc["answer"], start=.5, end=.95, interval=.05)

    return {"score": doc}

def aggregate_results(docs):

    accs = [doc["accuracy"] for doc in docs]
    accuracy = sum(accs) / len(accs) * 100.0
    accuracy = accuracy.mean().item()

    outputs = OrderedDict()
    outputs["Overall"] = accuracy

    tabulated_keys = ", ".join([_ for _ in outputs.keys()])
    tabulated_results = ", ".join([f"{_:.3f}" if isinstance(_, float) else str(_) for _ in outputs.values()])
    eval_logger.info(f"Tabulated results: {tabulated_keys}")
    eval_logger.info(f"Tabulated results: {tabulated_results}")
    outputs["Tabulated Keys"] = tabulated_keys
    outputs["Tabulated Results"] = tabulated_results

    return outputs
