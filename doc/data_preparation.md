# Data Preparation

Cambrian-P fine-tunes from Cambrian-S-7B stage 3 on three pieces:

| Piece | Source | Size |
|---|---|---|
| 1. VSI-590K (VQA + scene geometry) | [`nyu-visionx/vsi-590k`](https://huggingface.co/datasets/nyu-visionx/vsi-590k) | ~236 GB |
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

Build the training manifest (single jsonl read by the trainer):

```bash
python scripts/data/build_train_manifest.py \
    --vsi590k_jsonl    "$DATA_DIR/vsi-590k.jsonl"                          \
    --cambrian_s_jsonl "$VIPE_CAMBRIANS_DATA_ROOT/cambrian_s_3m_vqa.jsonl" \
    --vipe_lookup      "$VIPE_CAMBRIANS_RESULTS_ROOT/vipe_cambrians_with_vqa.json" \
    --source_config    data/vipe_source_config_high.json                   \
    --out              data/cambrianp_train.jsonl
```
