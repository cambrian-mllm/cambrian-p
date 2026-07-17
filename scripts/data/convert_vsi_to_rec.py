#!/usr/bin/env python3
"""Convert VSI-590K real-geometry video paths to reconstruction scene paths."""

import argparse
import json
import os
import os.path as osp
import sys
from collections import Counter


PATH_TEMPLATES = {
    "scannet": "processed_scannet_f/scans_train/{scene}",
    "scannetpp": "scannetpp/{scene}",
    "arkitscenes": "arkitscenes/Training/{scene}",
}


def get_source(entry, video):
    source = entry.get("source_dataset", entry.get("data_source", ""))
    source = str(source).lower()
    if source in PATH_TEMPLATES:
        return source

    parts = video.replace("\\", "/").lower().split("/")
    if "scannetpp" in parts:
        return "scannetpp"
    if "arkitscenes" in parts:
        return "arkitscenes"
    if "scannet" in parts or "processed_scannet_f" in parts:
        return "scannet"
    return None


def convert_entry(entry):
    video = entry.get("video", "")
    if not video:
        return entry, None

    source = get_source(entry, video)
    if source is None:
        return entry, None

    scene = osp.splitext(osp.basename(video.rstrip("/")))[0]
    converted = dict(entry)
    converted["video"] = PATH_TEMPLATES[source].format(scene=scene)
    converted["source_dataset"] = source
    converted["loading_type"] = "rec"
    return converted, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input VSI-590K JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if osp.abspath(args.input) == osp.abspath(args.output):
        parser.error("--input and --output must be different files")
    if osp.exists(args.output) and not args.force:
        print(f"Refusing to overwrite {args.output} (pass --force).", file=sys.stderr)
        return 2

    counts = Counter()
    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
    with open(args.input) as input_file, open(args.output, "w") as output_file:
        for line in input_file:
            if not line.strip():
                continue
            entry, source = convert_entry(json.loads(line))
            output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            counts[source or "unchanged"] += 1

    print(dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
