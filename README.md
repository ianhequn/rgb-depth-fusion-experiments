# RGB–Depth fusion experiments

Exploratory spatial question answering experiments extracted from the local WISE-Read research workspace. The code and observations are separated by protocol; neither result establishes that the model learned reliable geometric fusion.

## Experiments

| Route | Code | Evidence | Status |
| --- | --- | --- | --- |
| Two-image prompt | `experiments/prompt_fusion.py` | 8 questions, 2 physical scenes, 3 conditions (RGB, matched RGB+depth, mismatched RGB+depth), 24 calls | Development probe only |
| Sequential RGB → depth → RGB | `experiments/historical-sequential/` | Historical 300-question development evaluation, RGB LoRA and depth specialist LoRA | Archived, machine-specific scripts |

The prompt probe converts `.npy` depth maps to relative near-white renderings with `log1p` and per-image 1st/99th percentiles. The rendered values have **unverified physical units**. Its original run showed no visible object-answer rescue on the 8-question sample. See [`results/DECISION.md`](results/DECISION.md) and [`results/prompt_fusion.jsonl`](results/prompt_fusion.jsonl). This is qualitative evidence, not an accuracy estimate.

The historical sequential experiment uses separate LoRA adapters: RGB generates an initial state, depth generates complementary evidence, then RGB produces a final answer. The local report recorded RGB→Depth at 107/300 (35.67%), 3.33 percentage points below its RGB2 comparison, with rescue/harm 30/40 and a reported 95% CI of [-8.67, 2.33] percentage points. The archived scripts contain original absolute paths and depend on trained adapters that are **not** redistributed here. Treat the numbers as historical development results rather than an independently reproduced benchmark.

## Run the prompt probe

Requirements: Python, CUDA GPU, local Qwen2-VL-2B-Instruct weights, the Open3D development manifest, and the corresponding RGB and depth files. Install the dependencies in `requirements.txt` in a suitable CUDA environment. From the repository root:

```bash
python experiments/prompt_fusion.py \
  --manifest /path/to/dev_clean_300_manifest.jsonl \
  --rgb-dir /path/to/eval/rgb \
  --depth-dir /path/to/eval/depth \
  --model /path/to/Qwen2-VL-2B-Instruct \
  --output results/new_probe.jsonl
```

The script selects the first available row for each of four question types in `Campus` and `Residence`. The manifest must have `sample_uid`, `scene_id`, `task_type`, `question`, `reference_answer`, `rgb.path`, and `depth.path`. It resolves files by basename, as the original local experiment did. The mismatched condition uses a depth map from the other scene. Keep all source image/depth files aligned and verify provenance before interpretation.

## Data, scope, and next experiment

Datasets, images, model weights, checkpoints, training logs, human labels, and private workspace artifacts are not included. Source hashes in the result file support comparison with locally held data. The source workspace is preserved; this repository is a curated copy.

The next useful experiment is a controlled set of RGB-D object pairs with checked pixel alignment and depth direction. Compare RGB-only, depth-only, matched-depth, and wrong-depth with identical answer options, then evaluate on untouched physical scenes. The detailed decision is in `results/DECISION.md`.

No license is granted for external data or model weights. The repository code has no license file pending an ownership decision.
