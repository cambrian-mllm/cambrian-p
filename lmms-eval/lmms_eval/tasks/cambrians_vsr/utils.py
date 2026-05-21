import os
from pathlib import Path
from loguru import logger as eval_logger
import yaml

import datasets
from collections import OrderedDict


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
cache_name = "cambrians_vsr"


def doc_to_visual(doc):
    video_path = doc["video_path"]
    video_path = os.path.join(base_cache_dir, cache_name, video_path)
    doc["video_path"] = video_path
    return [doc["video_path"]]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):

    question = doc["question"].strip()
    options = doc["options"]

    question = question + "\nOptions:\n" + "\n".join(options) + "\nAnswer with the option's letter from the given choices directly."
    return question


def process_docs_10mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["type"] == "10mins")

def process_docs_30mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["type"] == "30mins")

def process_docs_60mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["type"] == "60mins")

def process_docs_120mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["type"] == "120mins")

def process_docs_240mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["type"] == "240mins")

def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def process_results(doc, results):

    doc["prediction"] = results[0]
    doc["accuracy"] = exact_match(fuzzy_matching(doc["prediction"]), doc["answer"])
    return {"score": doc}


def aggregate_results(docs):

    total, correct = 0, 0
    for doc in docs:
        total += 1
        correct += doc["accuracy"]
    accuracy = correct / total * 100.0 if total > 0 else 0.0
    eval_logger.info(f"Overall accuracy: {accuracy:.3f}")

    outputs = OrderedDict()
    outputs["Overall"] = accuracy

    tabulated_keys = ", ".join([_ for _ in outputs.keys()])
    tabulated_results = ", ".join([f"{_:.3f}" if isinstance(_, float) else _ for _ in outputs.values()])
    eval_logger.info(f"Tabulated results: {tabulated_keys}")
    eval_logger.info(f"Tabulated results: {tabulated_results}")
    outputs["Tabulated Keys"] = tabulated_keys
    outputs["Tabulated Results"] = tabulated_results

    return outputs
