#!/usr/bin/env python3
"""
Build the Cambrian-P training manifest from the three pieces a user downloads:

    1. VSI-590K       (HF: nyu-visionx/vsi-590k)         — VQA + scene geometry
    2. Cambrian-S 3M  (cambrian-mllm/cambrian-s)         — video VQA, broader corpus
    3. Cambrian-P-Data (HF: nyu-visionx/Cambrian-P-Data) — pose annotations + lookup

The output JSONL is NOT released — every user builds it locally so the manifest
stays in sync with whichever subset of Cambrian-S 3M they actually downloaded.

Pipeline (one pass):
    Cambrian-S 3M VQA JSONL
        ├─ filter to the 10 high-quality sources kept by source_config_high.json
        │  (Ego4d, favd, guiworld, k400_targz, k710, sharegpt4o, ssv2, star, tgif, timeit)
        ├─ keep every row whose video stem matches a ViPE pose annotation
        │  (so the pose loss has a supervision target for that sample)
        ├─ fill the remaining budget with random unmatched rows from the same
        │  source pool (seeded → reproducible)
        └─ stop at --cambrian_s_target rows (default 590,000)

    + VSI-590K JSONL (all rows, ~590K)
    → concat + seeded shuffle
    → data/cambrianp_train.jsonl  (~1.18M rows total)

Usage (paths match the env vars set in README §Data Preparation Quickstart):

    python scripts/data/build_train_manifest.py \\
        --vsi590k_jsonl      "$DATA_DIR/vsi-590k.jsonl" \\
        --cambrian_s_jsonl   "$VIPE_CAMBRIANS_DATA_ROOT/cambrian_s_3m_vqa.jsonl" \\
        --vipe_lookup        "$VIPE_CAMBRIANS_RESULTS_ROOT/vipe_cambrians_with_vqa.json" \\
        --source_config      data/vipe_source_config_high.json \\
        --out                data/cambrianp_train.jsonl

The `--cambrian_s_jsonl` input is Cambrian-S's own VQA manifest. Refer to their
data instructions for the exact filename; it's the JSONL with one row per video
sample, `video` field as a path stem like `<source>/<uid>.mp4`.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import sys
from typing import Dict, List, Tuple


def stem(path: str) -> str:
    return osp.splitext(osp.basename(path))[0]


def source_from_video_path(video: str) -> str:
    """Best-effort source extraction from a video path. Matches the internal
    builder's convention: paths look like `<prefix>/<source>/<uid>.mp4`, so the
    immediate parent of the basename is the source."""
    parts = video.replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return ""


def load_source_config(path: str) -> set:
    """Return the set of sources we keep (where the json value is truthy)."""
    with open(path) as f:
        cfg = json.load(f)
    keep = {k for k, v in cfg.items() if v and not k.startswith("_")}
    return keep


def load_vipe_stems(path: str) -> Tuple[set, set]:
    """Read the ViPE lookup JSON (~516 MB) once and return two indices:
    by_source_stem and by_stem (for fallback when the source label disagrees)."""
    with open(path) as f:
        entries = json.load(f)
    by_source_stem = set()
    by_stem = set()
    for e in entries:
        src = e.get("source", "")
        v = e.get("video", "") or e.get("rgb_path", "")
        s = stem(v)
        if s:
            by_stem.add(s)
            if src:
                by_source_stem.add((src, s))
    return by_source_stem, by_stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsi590k_jsonl", required=True,
                    help="VSI-590K JSONL (HF: nyu-visionx/vsi-590k).")
    ap.add_argument("--cambrian_s_jsonl", required=True,
                    help="Cambrian-S 3M VQA JSONL (from the Cambrian-S data repo).")
    ap.add_argument("--vipe_lookup", required=True,
                    help="$VIPE_CAMBRIANS_RESULTS_ROOT/vipe_cambrians_with_vqa.json (HF: Cambrian-P-Data).")
    ap.add_argument("--source_config", default="data/vipe_source_config_high.json",
                    help="Source filter (ships in repo).")
    ap.add_argument("--out", default="data/cambrianp_train.jsonl",
                    help="Output JSONL.")
    ap.add_argument("--cambrian_s_target", type=int, default=590_000,
                    help="Total Cambrian-S rows to keep (matched + filled).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite --out if it exists.")
    args = ap.parse_args()

    if osp.exists(args.out) and not args.force:
        print(f"Refusing to overwrite {args.out} (pass --force).", file=sys.stderr)
        return 2

    sources_keep = load_source_config(args.source_config)
    print(f"[source_config] keep ({len(sources_keep)}): {sorted(sources_keep)}", file=sys.stderr)

    print(f"[vipe] indexing {args.vipe_lookup}…", file=sys.stderr)
    by_source_stem, by_stem = load_vipe_stems(args.vipe_lookup)
    print(f"[vipe] {len(by_stem):,} unique stems, {len(by_source_stem):,} (source,stem) pairs", file=sys.stderr)

    # Pass 1: scan Cambrian-S JSONL, split into (in-source-pool & matched) vs
    # (in-source-pool & unmatched). Discard anything outside the source pool.
    matched, unmatched = [], []
    n_total = n_in_pool = 0
    with open(args.cambrian_s_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            row = json.loads(line)
            video = row.get("video", "")
            if not video:
                continue
            src = source_from_video_path(video)
            if src not in sources_keep:
                continue
            n_in_pool += 1
            s = stem(video)
            if (src, s) in by_source_stem or s in by_stem:
                matched.append(line)
            else:
                unmatched.append(line)

    print(f"[cambrian_s] read={n_total:,}  in_source_pool={n_in_pool:,}  "
          f"matched={len(matched):,}  unmatched={len(unmatched):,}", file=sys.stderr)

    if args.cambrian_s_target < len(matched):
        raise ValueError(
            f"--cambrian_s_target={args.cambrian_s_target} < matched={len(matched)}; "
            "this builder is designed to keep every pose-matched row.")

    need_unmatched = args.cambrian_s_target - len(matched)
    if need_unmatched > len(unmatched):
        raise ValueError(
            f"need {need_unmatched:,} unmatched rows to reach target "
            f"{args.cambrian_s_target:,}, but only {len(unmatched):,} available. "
            "Either lower --cambrian_s_target or download more Cambrian-S sources.")

    rng = random.Random(args.seed)
    fill = rng.sample(unmatched, need_unmatched)
    cs_rows = matched + fill
    rng.shuffle(cs_rows)
    print(f"[cambrian_s] kept_matched={len(matched):,}  filled={need_unmatched:,}  "
          f"total={len(cs_rows):,}", file=sys.stderr)

    # Pass 2: stream VSI-590K (don't load into memory if avoidable).
    n_vsi = 0
    with open(args.vsi590k_jsonl) as f:
        vsi_rows = [ln.strip() for ln in f if ln.strip()]
    n_vsi = len(vsi_rows)
    print(f"[vsi-590k] rows={n_vsi:,}", file=sys.stderr)

    # Concat + shuffle (seeded). One global mix so per-step batches see both
    # distributions; matches the convention used to train the released ckpts.
    all_rows = cs_rows + vsi_rows
    rng.shuffle(all_rows)

    tmp = args.out + ".tmp"
    os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        for ln in all_rows:
            f.write(ln)
            f.write("\n")
    os.replace(tmp, args.out)

    stats = {
        "out": args.out,
        "rows": len(all_rows),
        "cambrian_s_matched": len(matched),
        "cambrian_s_filled": need_unmatched,
        "vsi590k_rows": n_vsi,
        "seed": args.seed,
        "source_config_keep": sorted(sources_keep),
    }
    print(json.dumps(stats, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
