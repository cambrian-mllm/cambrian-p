import datetime
import json
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

# Get the Hugging Face cache directory
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)

# Load the default template YAML config
with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def tomato_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """
    Extract the video path from the document.
    
    Args:
        doc: A document instance from the dataset
        lmms_eval_specific_kwargs: Additional parameters
        
    Returns:
        A list containing the video path
    """
    cache_dir = os.path.join(base_cache_dir, cache_name)
    
    # Get the demonstration type (human, object, or simulated)
    demo_type = doc["demonstration_type"]
    
    # Get the video key and add .mp4 extension
    video_key = doc["key"] + ".mp4"
    
    # Construct the video path
    video_path = os.path.join(cache_dir, "videos", demo_type, video_key)
    
    if os.path.exists(video_path):
        return [video_path]
    else:
        # Try alternative path
        alt_path = os.path.join(cache_dir, demo_type, video_key)
        if os.path.exists(alt_path):
            return [alt_path]
        
        eval_logger.error(f"Video path: {video_path} does not exist, please check.")
        return [video_path]  # Return the original path even if it doesn't exist


def tomato_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Format the question with options for the model.
    
    Args:
        doc: A document instance from the dataset
        lmms_eval_specific_kwargs: Additional parameters
        
    Returns:
        The formatted question text
    """
    option_prompt = ""
    option_list = doc["options"]
    option_letters = string.ascii_uppercase
    
    for char_index, option in enumerate(option_list):
        option_letter = option_letters[char_index]
        option_prompt += f"({option_letter}) {option}\n"

    full_text = "Question: " + doc["question"] + "\nOptions:\n" + option_prompt
    
    if lmms_eval_specific_kwargs and "post_prompt" in lmms_eval_specific_kwargs:
        full_text += lmms_eval_specific_kwargs["post_prompt"]
    
    return full_text


def mcq_acc(answer, pred):
    """
    Calculate the accuracy of a multiple-choice question.
    
    Args:
        answer: The correct answer
        pred: The predicted answer
        
    Returns:
        1 if the prediction is correct, 0 otherwise
    """
    periodStrip = re.compile("(?!<=\\d)(\\.(?!\\d))")
    commaStrip = re.compile("(\\d)(\\,)(\\d)")
    punct = [
        ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", 
        "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"
    ]

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) is not None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(answer):
        answer = answer.strip()

        # Handle "Answer: (X) text" or "Answer: X" format
        answer_prefix = re.match(r"^(?:answer|the answer is)\s*[:.]?\s*(.+)$", answer, re.IGNORECASE)
        if answer_prefix:
            answer = answer_prefix.group(1).strip()

        # Check for option letter pattern: "A", "A.", "A)", "(A)", "(A) text", "A. text"
        option_regex = re.compile(r"^\(?([A-G])\)?[\.|\)]?\s*(.*)$", re.IGNORECASE)
        match = option_regex.match(answer.strip())

        if match:
            # If matched, return the option letter in uppercase
            return match.group(1).upper()
        else:
            # Process the answer text
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            answer = answer.strip()
            answer = processPunctuation(answer)
            answer = answer.strip("'")
            answer = answer.strip('"')
            answer = answer.strip(")")
            answer = answer.strip("(")
            answer = answer.strip().lower()

            # Try to find any single letter (A-F) in the processed answer
            letter_match = re.search(r"\b([A-G])\b", answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()

            return answer

    pred = process(pred)
    answer = process(answer)

    if pred == answer:
        score = 1
    else:
        score = 0

    return score


def tomato_process_results(doc, results):
    """
    Process the model's response for evaluation.
    
    Args:
        doc: A document instance from the dataset
        results: The model's prediction
        
    Returns:
        A dictionary with the evaluation results
    """
    pred = results[0]

    # Get the ground truth answer index
    answer_index = doc["answer"]
    
    # Calculate the ground truth option letter
    option_letters = string.ascii_uppercase
    gt_option_letter = option_letters[answer_index]

    # Calculate the score
    score = mcq_acc(gt_option_letter, pred)

    data_dict = {
        "pred_answer": pred, 
        "gt_answer": gt_option_letter, 
        "score": score,
        "motion_type": doc["motion_type"]
    }

    return {"tomato_accuracy": data_dict}


def tomato_doc_to_answer(doc):
    """
    Extract the answer from the document.
    
    Args:
        doc: A document instance from the dataset
        
    Returns:
        The correct answer option
    """
    answer_index = doc["answer"]
    option_letters = string.ascii_uppercase
    return f"({option_letters[answer_index]}) {doc['options'][answer_index]}"


def tomato_aggregate_results(results):
    """
    Aggregate the evaluation results.
    
    Args:
        results: A list of evaluation results
        
    Returns:
        The overall accuracy score
    """
    total_answered = 0
    total_correct = 0

    # Also track results by motion type
    motion_type_results = defaultdict(lambda: {"total": 0, "correct": 0})

    # NOTE: count ALL samples in the denominator. Previously this skipped
    # samples with empty pred_answer, which silently inflated accuracy for
    # checkpoints that fail to generate (e.g. ex136-5 had 506/1484 empties
    # and reported 23.99% instead of the honest 18.38%). An empty model
    # output is a wrong answer, not an excused absence.
    for result in results:
        total_answered += 1
        total_correct += result["score"]

        # Track by motion type if available
        if "motion_type" in result:
            motion_type = result["motion_type"]
            motion_type_results[motion_type]["total"] += 1
            motion_type_results[motion_type]["correct"] += result["score"]
    
    # Print results by motion type for detailed analysis
    for motion_type, stats in motion_type_results.items():
        if stats["total"] > 0:
            accuracy = 100 * stats["correct"] / stats["total"]
            eval_logger.info(f"Motion type '{motion_type}' accuracy: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
    
    return 100 * total_correct / total_answered if total_answered > 0 else 0