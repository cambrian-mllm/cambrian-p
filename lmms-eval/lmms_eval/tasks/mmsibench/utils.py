import io
import logging
import re
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
from PIL import Image

eval_logger = logging.getLogger("lmms-eval")

# Question type categories for MMSI-Bench
MMSI_QUESTION_TYPES = [
    "Positional Relationship (Obj.-Obj.)",
    "Positional Relationship (Cam.-Obj.)",
    "Positional Relationship (Cam.-Cam.)",
    "Positional Relationship (Obj.-Reg.)",
    "Positional Relationship (Cam.-Reg.)",
    "Positional Relationship (Reg.-Reg.)",
    "Attribute (Meas.)",
    "Attribute (Appr.)",
    "Motion (Obj.)",
    "Motion (Cam.)",
]


def msr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert MMSI-Bench document to text prompt."""
    question = doc["question"].strip()
    
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    
    return question


def msr_doc_to_visual(doc):
    """Load images from MMSI-Bench document (binary format from parquet)."""
    image_list = []
    for img_data in doc["images"]:
        # Handle both bytes and already-loaded images
        if isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data))
        elif hasattr(img_data, 'convert'):
            # Already a PIL Image
            image = img_data
        else:
            # Try treating as bytes-like
            image = Image.open(io.BytesIO(bytes(img_data)))
        image = image.convert("RGB")
        image_list.append(image)
    return image_list


def extract_single_choice_with_word_boundary(pred, gt):
    """Extract single choice answer from prediction."""
    # Pattern 1: ``answer``
    pattern_1 = r"``([^`]*)``"
    match = re.search(pattern_1, pred)
    if match:
        pred = match.group(1)
    
    # Pattern 2: `answer`
    pattern_2 = r"`([^`]*)`"
    match = re.search(pattern_2, pred)
    if match:
        pred = match.group(1)
    
    # Pattern 3: {answer}
    pattern_add = r"\{([^}]*)\}"
    match = re.search(pattern_add, pred)
    if match:
        pred = match.group(1)
    
    # Pattern 4: Single letter A-D with word boundary
    pattern_3 = r"\b[A-D]\b(?!\s[a-zA-Z])"
    match = re.search(pattern_3, pred)
    if match:
        pred = match.group()
    else:
        return None
    
    answer = gt.lower().replace("\n", " ").strip()
    predict = pred.lower().replace("\n", " ").strip()
    
    try:
        if answer == predict[0]:
            return 1.0
        elif predict[0] == "(" and answer == predict[1]:
            return 1.0
        elif predict[0:7] == "option " and answer == predict[7]:
            return 1.0
        elif predict[0:14] == "the answer is " and answer == predict[14]:
            return 1.0
    except Exception as e:
        return 0.0
    
    return 0.0


def msr_process_results(doc, results):
    """
    Process model results for a single MMSI-Bench sample.
    
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"]
    score = extract_single_choice_with_word_boundary(pred, gt)
    category = doc["question_type"]
    l2_category = doc["question_type"]
    
    if score is None:
        return {
            category: {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "can not find answer"},
            "average": {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "can not find answer"},
            "mmsi_score": {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "category": category}
        }
    
    return {
        category: {"question_id": doc["id"], "l2_category": l2_category, "score": score},
        "average": {"question_id": doc["id"], "l2_category": l2_category, "score": score},
        "mmsi_score": {"question_id": doc["id"], "l2_category": l2_category, "score": score, "category": category}
    }


def msr_aggregate_results(results):
    """
    Aggregate results across all MMSI-Bench samples.
    
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    l2_category_scores = defaultdict(list)
    
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)
    
    l2_category_avg_score = {}
    for l2_category, scores in l2_category_scores.items():
        avg_score = sum(scores) / len(scores)
        l2_category_avg_score[l2_category] = avg_score
        eval_logger.info(f"{l2_category}: {avg_score:.2f}")
    
    all_scores = [score for scores in l2_category_scores.values() for score in scores]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    return avg_score


def mmsi_aggregate_results(results):
    """
    Aggregate results for MMSI-Bench with detailed breakdown.
    Returns an OrderedDict with overall score and per-category scores.
    """
    category_scores = defaultdict(list)
    
    for result in results:
        score = result["score"]
        category = result.get("category", result.get("l2_category", "unknown"))
        category_scores[category].append(score)
    
    output = OrderedDict()
    
    # Calculate per-category averages
    category_avg = {}
    for category, scores in category_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        category_avg[category] = avg
        eval_logger.info(f"{category}: {avg * 100:.2f}%")
    
    # Overall average
    all_scores = [s for scores in category_scores.values() for s in scores]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    output["overall"] = overall * 100.0
    
    # Add per-category results in order
    for qtype in MMSI_QUESTION_TYPES:
        if qtype in category_avg:
            # Clean key name for output
            clean_key = qtype.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").replace("-", "_")
            output[clean_key] = category_avg[qtype] * 100.0
    
    # Create tabulated output
    tabulated_keys = ", ".join([str(k) for k in output.keys()])
    tabulated_results = ", ".join([f"{v:.3f}" for v in output.values()])
    
    eval_logger.info(f"Tabulated keys: {tabulated_keys}")
    eval_logger.info(f"Tabulated results: {tabulated_results}")
    
    output["tabulated_keys"] = tabulated_keys
    output["tabulated_results"] = tabulated_results
    
    return output