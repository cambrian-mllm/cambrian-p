import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd

import datasets
from collections import OrderedDict, defaultdict

TASK_TYPES = ['navigation/object_retrieval', 'navigation/object_retrieval_image', 'navigation/room_to_room', 'navigation/room_to_room_image', 'perception/information_retrieval/factual_recall', 'perception/information_retrieval/sequence_recall', 'perception/information_retrieval/temporal_distance', 'perception/tracking', 'reasoning/causal', 'reasoning/counterfactual', 'reasoning/predictive', 'reasoning/spatial/layout', 'reasoning/spatial/proximity', 'reasoning/spatial/relationship', 'reasoning/temporal/duration', 'reasoning/temporal/frequency', 'reasoning/temporal/prerequisites', 'summarization/compare_and_contrast', 'summarization/key_events_objects', 'summarization/temporal_sequencing']

# Comment out the body of this function to perform blind evaluation.
def hourvideo_doc_to_visual(doc):
    video_path = os.path.join("/data/cambrian-s/Ego4D/v4/full_scale/", doc["video_uid"] + ".mp4")
    if doc["task"] in ["navigation/room_to_room_image", "reasoning/spatial/layout", "navigation/object_retrieval_image"]:
        image_paths = []
        for _ in range(5):
            image_path = os.path.join("/data/cambrian-s/huggingface/HourVideo/v1.0_release/", doc[f"answer_{_+1}"])
            image_paths.append(image_path)
        return [video_path] + image_paths
    else:
        return [video_path]

def hourvideo_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") or "Answer with the option's letter from the given choices directly."

    if doc["task"] in ["navigation/room_to_room_image", "reasoning/spatial/layout", "navigation/object_retrieval_image"]:
        question = ["", pre_prompt + "\n" + question + "\n" + "Options:"]
        for _ in range(5):
            question.append("\n" + chr(65 + _) + ". ")
            question.append("")
        question.append(post_prompt)
        return question
    else:
        options = "Options:\n" + doc["mcq_test"]
        return "\n".join([pre_prompt, question, options, post_prompt])

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    if os.getenv("CAMBRIAN_BLIND_EVAL", None) == "1":
        eval_logger.info(f"CAMBRIAN_BLIND_EVAL detected, HourVideo dataset will be filtered to remove the tasks that use images as choices.")
        dataset = dataset.filter(lambda x: x["task"] not in ["navigation/room_to_room_image", "reasoning/spatial/layout", "navigation/object_retrieval_image"])
    return dataset

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.


def hourvideo_process_results(doc, results):
    
    doc["prediction"] = results[0]
    doc["accuracy"] = exact_match(fuzzy_matching(doc["prediction"]), fuzzy_matching(doc["correct_answer_label"]))

    return {"hourvideo_score": doc}

def hourvideo_aggregate_results(docs):

    results = defaultdict(lambda: {"correct": 0, "count": 0, "accuracy": 0.0})
    for doc in docs:
        results[doc['task']]["count"] += 1
        results[doc['task']]["correct"] += doc["accuracy"]
        
        results["overall"]["count"] += 1
        results["overall"]["correct"] += doc["accuracy"]
    
    results["overall"]["accuracy"] = results["overall"]["correct"] / results["overall"]["count"]
    for task_type in TASK_TYPES:
        if results[task_type]["count"] > 0:
            results[task_type]['accuracy'] = results[task_type]['correct'] / results[task_type]['count']
        else:
            results[task_type]['accuracy'] = 0.0

    outputs = OrderedDict()
    outputs["overall"] = results["overall"]["accuracy"] * 100.0
    for task_type in TASK_TYPES:
        outputs[task_type] = results[task_type]['accuracy'] * 100.0

    tabulated_keys = ", ".join([_ for _ in outputs.keys()])
    tabulated_results = ", ".join([f"{_:.3f}" for _ in outputs.values()])
    eval_logger.info(f"Tabulated results: {tabulated_keys}")
    eval_logger.info(f"Tabulated results: {tabulated_results}")
    outputs["tabulated_keys"] = tabulated_keys
    outputs["tabulated_results"] = tabulated_results

    return outputs
