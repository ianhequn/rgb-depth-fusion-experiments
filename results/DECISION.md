# RGB–depth fusion smoke and next experiment

## Scope

- Historical Open3D development records only: 8 questions, 2 physical scenes, 4 question types per scene.
- Qwen2-VL-2B-Instruct, no training, greedy decoding, `max_new_tokens=32`, 12 inference calls per scene / 24 total.
- Inputs: RGB only; RGB plus matched depth rendering; RGB plus depth rendering from the other physical scene.
- Depth is rendered with per-image log and percentile scaling; pixel brightness encodes relative proximity, not a calibrated distance in metres.
- Source RGB and depth file hashes, prompts, answers and source scene IDs are in `rgb_depth_fusion_smoke.jsonl`.
- Local GPU peak allocated memory: 4.21 GiB.

## Observations

- No visible object-answer rescue among these eight questions.
- Campus `front_choice`: RGB names the reference object (large textured wall). Matched depth changes the object to floating rock; mismatched depth retains the wall.
- Residence `behind_choice`: RGB names the reference object (wall-mounted air conditioner). Both matched and mismatched depth change the object to the window grid. This cannot be attributed to correct depth content.
- Other cases retain the same object answer or change only wording.
- These are qualitative observations on a deliberately tiny development sample, not an accuracy estimate or evidence of a model-wide effect.

## Decision

The direct two-image prompt is a runnable interface but has not demonstrated semantic depth use. Before training any fusion module, create controlled RGB-D object pairs with verified pixel alignment and depth direction. The test must separate (a) identifying the referent in RGB, (b) reading its depth at matched pixels, and (c) comparing two referents. Use matched-depth, wrong-depth, depth-only, and RGB-only conditions with the same answer options. A genuine geometry-use signal should follow the matched depth under depth-changing counterfactuals and reject mismatched depth.

If that interface passes on development cases, compare a lightweight geometry branch or specialist adapter against prompt-only RGB+depth. Train on scene-disjoint controlled pairs, then evaluate on untouched physical scenes. Do not treat these eight development cases as a test set.
