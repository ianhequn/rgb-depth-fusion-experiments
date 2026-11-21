"""Development-only RGB/depth prompt fusion probe for Open3D QA.

This is an inference interface experiment, not calibrated geometry reasoning.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

TASKS = ("behind_choice", "front_choice", "left_choice", "above_choice")
SCENES = ("Campus", "Residence")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_depth(path):
    raw = np.load(path).astype(np.float32)
    valid = np.isfinite(raw) & (raw > 0)
    if not valid.any():
        raise ValueError(f"No positive finite depth pixels: {path}")
    transformed = np.log1p(np.where(valid, raw, 0))
    lo, hi = np.percentile(transformed[valid], [1, 99])
    scaled = 1 - np.clip((transformed - lo) / max(hi - lo, 1e-6), 0, 1)
    rendered = np.where(valid, (scaled * 255).astype(np.uint8), 0)
    return Image.fromarray(rendered, "L").convert("RGB")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--rgb-dir", type=Path, required=True)
    p.add_argument("--depth-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/prompt_fusion.jsonl"))
    return p.parse_args()


def main():
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    available = [r for r in rows if (args.rgb_dir / Path(r["rgb"]["path"]).name).exists()
                 and (args.depth_dir / Path(r["depth"]["path"]).name).exists()]
    chosen = [next(r for r in available if r["scene_id"] == scene and r["task_type"] == task)
              for scene in SCENES for task in TASKS]
    if len({r["sample_uid"] for r in chosen}) != len(chosen):
        raise ValueError("Selected rows are not unique")

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True,
                                              max_pixels=28 * 28 * 256)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=torch.float16, device_map="cuda",
        local_files_only=True, attn_implementation="sdpa").eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh, torch.inference_mode():
        for row in chosen:
            rgb_path = args.rgb_dir / Path(row["rgb"]["path"]).name
            depth_path = args.depth_dir / Path(row["depth"]["path"]).name
            wrong_row = next(r for r in available if r["scene_id"] != row["scene_id"])
            wrong_path = args.depth_dir / Path(wrong_row["depth"]["path"]).name
            rgb = Image.open(rgb_path).convert("RGB")
            depth = render_depth(depth_path)
            wrong_depth = render_depth(wrong_path)
            for condition, images in (
                ("RGB", [rgb]),
                ("RGB_PLUS_MATCHED_DEPTH", [rgb, depth]),
                ("RGB_PLUS_MISMATCHED_DEPTH", [rgb, wrong_depth]),
            ):
                content = [{"type": "image", "image": im} for im in images]
                if len(images) == 2:
                    prompt = ("Image 1 is RGB. Image 2 is a relative depth rendering: "
                              "brighter valid pixels are closer; black may be invalid. "
                              "The two images may not correspond. Use depth only when its "
                              "objects align with RGB.\n")
                else:
                    prompt = "The image is RGB.\n"
                prompt += row["question"] + "\nAnswer concisely."
                content.append({"type": "text", "text": prompt})
                template = processor.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False,
                    add_generation_prompt=True)
                inputs = processor(text=[template], images=images, return_tensors="pt")
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                answer = processor.decode(output[0, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip()
                record = {
                    "status": "DEVELOPMENT_ONLY_EXPLORATORY", "sample_uid": row["sample_uid"],
                    "scene_id": row["scene_id"], "task_type": row["task_type"],
                    "question": row["question"], "reference_answer": row["reference_answer"],
                    "condition": condition, "prediction": answer,
                    "rgb_sha256": sha256(rgb_path), "depth_sha256": sha256(depth_path),
                    "wrong_depth_sha256": sha256(wrong_path),
                    "wrong_depth_scene_id": wrong_row["scene_id"],
                    "depth_render": "log1p_per_image_p1_p99_near_white_invalid_black",
                    "depth_units": "UNVERIFIED", "model": args.model.name,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "max_new_tokens": 32, "do_sample": False,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                print(row["sample_uid"], condition, repr(answer), flush=True)
    print("OUT", args.output, "PEAK_GB", round(torch.cuda.max_memory_allocated() / 1024**3, 2))


if __name__ == "__main__":
    main()
