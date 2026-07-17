# Data Preparation

Cambrian-P fine-tunes from Cambrian-S-7B stage 3 on three pieces:

| Piece | Source | Size |
|---|---|---|
| 1. VSI-590K (VQA annotations + RGB videos) | [`nyu-visionx/vsi-590k`](https://huggingface.co/datasets/nyu-visionx/vsi-590k) | ~236 GB |
| 2. Cambrian-S 3M videos | [`nyu-visionx/Cambrian-S-3M`](https://huggingface.co/datasets/nyu-visionx/Cambrian-S-3M) | per-source |
| 3. Cambrian-P pose annotations | [`nyu-visionx/Cambrian-P-Data`](https://huggingface.co/datasets/nyu-visionx/Cambrian-P-Data) | ~850 MiB |

## Quickstart

```bash
export DATA_DIR=/path/to/vsi-590k
export VIPE_CAMBRIANS_RESULTS_ROOT=/path/to/cambrian_p_pose
export VIPE_CAMBRIANS_DATA_ROOT=/path/to/cambrian_s_3m

huggingface-cli download nyu-visionx/vsi-590k        --repo-type dataset --local-dir "$DATA_DIR"
huggingface-cli download nyu-visionx/Cambrian-P-Data --repo-type dataset --local-dir "$VIPE_CAMBRIANS_RESULTS_ROOT"

( cd "$VIPE_CAMBRIANS_RESULTS_ROOT" && for t in pose/*.tar; do tar xf "$t"; done )

python scripts/data/build_vipe_lookup.py \
    --results_root "$VIPE_CAMBRIANS_RESULTS_ROOT" \
    --out          "$VIPE_CAMBRIANS_RESULTS_ROOT/vipe_cambrians_with_vqa.json"

ln -sf "$VIPE_CAMBRIANS_RESULTS_ROOT/vipe_cambrians_with_vqa.json" data/vipe_cambrians_with_vqa.json
```

## Real-geometry data

The ScanNet, ScanNet++, and ARKitScenes videos in VSI-590K are loaded as regular
VQA videos. Reconstruction supervision requires the original datasets.

For ScanNet, use the original [CUT3R preprocessing](https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md)
without modification:

```bash
python preprocess_scannet.py \
    --scannet_dir /path/to/raw/scannet \
    --output_dir "$DATA_DIR/processed_scannet_f"
python generate_set_scannet.py \
    --root "$DATA_DIR/processed_scannet_f" \
    --splits scans_test scans_train \
    --max_interval 150 \
    --num_workers 8
```

For ScanNet++ and ARKitScenes, the provided scripts follow the CUT3R data format
but process all available frames instead of requiring preselected pairs:

```bash
export CUT3R_ROOT=/path/to/CUT3R
export PYTHONPATH="$CUT3R_ROOT:$CUT3R_ROOT/src:$PYTHONPATH"

python scripts/data/preprocess_scannetpp.py \
    --scannetpp_dir /path/to/raw/scannetpp \
    --output_dir "$DATA_DIR/scannetpp"

python scripts/data/preprocess_arkitscenes.py \
    --arkitscenes_dir /path/to/raw/arkitscenes \
    --output_dir "$DATA_DIR/arkitscenes" \
    --num_workers 8
```

These scripts create `scene_metadata_all.npz`; no precomputed-pair metadata is
required. Convert the corresponding VSI rows to reconstruction paths before
building the training manifest:

```bash
python scripts/data/convert_vsi_to_rec.py \
    --input "$DATA_DIR/vsi-590k.jsonl" \
    --output "$DATA_DIR/vsi-590k-rec.jsonl"
```

Build the training manifest (single jsonl read by the trainer):

```bash
python scripts/data/build_train_manifest.py \
    --vsi590k_jsonl    "$DATA_DIR/vsi-590k-rec.jsonl"                          \
    --cambrian_s_jsonl "$VIPE_CAMBRIANS_DATA_ROOT/cambrian_s_3m_vqa.jsonl" \
    --vipe_lookup      "$VIPE_CAMBRIANS_RESULTS_ROOT/vipe_cambrians_with_vqa.json" \
    --source_config    data/vipe_source_config_high.json                   \
    --out              data/cambrianp_train.jsonl
```
