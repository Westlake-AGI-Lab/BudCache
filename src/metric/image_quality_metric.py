from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.metric._metric_io import (
    discover_strategy_dirs,
    is_image_file,
    print_metric_table,
    save_per_metric,
    update_top_summary,
    write_top_meta,
)
from src.metric.plot_metrics import plot_all
from src.util import model_root


def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def build_image_index(
    strategy_dirs: dict[str, Path],
) -> tuple[dict[str, dict[str, Path]], list[int]]:
    reference_strategy = next(iter(strategy_dirs))
    reference_prefix = "p00000_s"
    sample_indices = sorted(
        int(path.stem[len(reference_prefix):])
        for path in strategy_dirs[reference_strategy].iterdir()
        if is_image_file(path) and path.stem.startswith(reference_prefix)
    )
    if not sample_indices:
        raise ValueError(
            f"{reference_strategy} does not contain files like p00000_s00.png"
        )
    strategy_to_image_map: dict[str, dict[str, Path]] = {}
    for strategy, strategy_dir in strategy_dirs.items():
        images = sorted(p for p in strategy_dir.iterdir() if is_image_file(p))
        if not images:
            raise ValueError(f"No images found in {strategy_dir}")
        strategy_to_image_map[strategy] = {p.stem: p for p in images}
    return strategy_to_image_map, sample_indices


def evaluate_image_reward(
    output_dir: Path,
    metric_dir: Path,
    strategy_to_image_map: dict[str, dict[str, Path]],
    sample_indices: list[int],
    prompts: list[str],
    model_root: str,
) -> None:
    import ImageReward as RM

    ckpt = os.path.join(model_root, "ImageReward/ImageReward.pt")
    model = RM.load(ckpt)
    records: list[dict] = []
    by_strategy: dict[str, list[float]] = {}
    for strategy, image_map in strategy_to_image_map.items():
        scores: list[float] = []
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"ImageReward:{strategy}")):
            for sample_idx in sample_indices:
                sample_id = f"p{prompt_idx:05d}_s{sample_idx:02d}"
                if sample_id not in image_map:
                    raise ValueError(f"{strategy} missing {sample_id}")
                score = float(model.score(prompt, str(image_map[sample_id])))
                scores.append(score)
                records.append({
                    "strategy": strategy,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "image": str(image_map[sample_id]),
                    "score": score,
                })
        by_strategy[strategy] = scores
    print_metric_table("ImageReward", by_strategy)
    save_per_metric(metric_dir, "imagereward.json", "imagereward", ckpt, records, by_strategy)
    update_top_summary(output_dir, "imagereward", by_strategy)
    del model
    torch.cuda.empty_cache()


def evaluate_hpsv2(
    output_dir: Path,
    metric_dir: Path,
    strategy_to_image_map: dict[str, dict[str, Path]],
    sample_indices: list[int],
    prompts: list[str],
    hps_version: str,
) -> None:
    import hpsv2

    records: list[dict] = []
    by_strategy: dict[str, list[float]] = {}
    for strategy, image_map in strategy_to_image_map.items():
        scores: list[float] = []
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"HPSv2:{strategy}")):
            for sample_idx in sample_indices:
                sample_id = f"p{prompt_idx:05d}_s{sample_idx:02d}"
                if sample_id not in image_map:
                    raise ValueError(f"{strategy} missing {sample_id}")
                score = float(
                    hpsv2.score(
                        str(image_map[sample_id]),
                        prompt,
                        hps_version=hps_version,
                    )[0]
                )
                scores.append(score)
                records.append({
                    "strategy": strategy,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "image": str(image_map[sample_id]),
                    "score": score,
                })
        by_strategy[strategy] = scores
    print_metric_table("HPSv2", by_strategy)
    save_per_metric(metric_dir, "hps.json", f"hps_{hps_version}", f"HPSv2-{hps_version}", records, by_strategy)
    write_top_meta(output_dir, {"hps_version": hps_version})
    update_top_summary(output_dir, "hps", by_strategy)


@torch.no_grad()
def clip_score_one(
    image_path: str,
    prompt: str,
    model: CLIPModel,
    processor: CLIPProcessor,
) -> float:
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to("cuda")
    text_inputs = processor(
        text=[prompt],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=processor.tokenizer.model_max_length,
    ).to("cuda")
    image_features = model.get_image_features(**image_inputs)
    text_features = model.get_text_features(**text_inputs)
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    return float((image_features * text_features).sum(dim=-1).item())


def evaluate_clip(
    output_dir: Path,
    metric_dir: Path,
    strategy_to_image_map: dict[str, dict[str, Path]],
    sample_indices: list[int],
    prompts: list[str],
    model_root: str,
) -> None:
    ckpt = os.path.join(model_root, "openai-clip-vit-large-patch14")
    model = CLIPModel.from_pretrained(ckpt).to("cuda").eval()
    processor = CLIPProcessor.from_pretrained(ckpt)
    records: list[dict] = []
    by_strategy: dict[str, list[float]] = {}
    for strategy, image_map in strategy_to_image_map.items():
        scores: list[float] = []
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"CLIPScore:{strategy}")):
            for sample_idx in sample_indices:
                sample_id = f"p{prompt_idx:05d}_s{sample_idx:02d}"
                if sample_id not in image_map:
                    raise ValueError(f"{strategy} missing {sample_id}")
                score = clip_score_one(str(image_map[sample_id]), prompt, model, processor)
                scores.append(score)
                records.append({
                    "strategy": strategy,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "image": str(image_map[sample_id]),
                    "score": score,
                })
        by_strategy[strategy] = scores
    print_metric_table("CLIPScore", by_strategy)
    save_per_metric(metric_dir, "clip.json", "clip", ckpt, records, by_strategy)
    update_top_summary(output_dir, "clip", by_strategy)
    del model, processor
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("prompts_file")
    parser.add_argument("--hps_version", default="v2.1")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["imagereward", "hps", "clip"],
        choices=["imagereward", "hps", "clip"],
    )
    args = parser.parse_args()

    model_dir = model_root()
    prompts = load_prompts(args.prompts_file)
    output_root = Path(args.output_dir)
    metric_dir = output_root / "0_metric"
    metric_dir.mkdir(parents=True, exist_ok=True)

    strategy_dirs = discover_strategy_dirs(output_root)
    if not strategy_dirs:
        raise ValueError(f"No valid image folders found in {args.output_dir}")
    strategy_to_image_map, sample_indices = build_image_index(strategy_dirs)

    print(f"Detected strategies: {list(strategy_to_image_map.keys())}")
    print(f"Detected samples_per_prompt: {len(sample_indices)}")
    for strategy, image_map in strategy_to_image_map.items():
        print(f"{strategy}: {len(image_map)} images")

    write_top_meta(output_root, {
        "strategies": list(strategy_to_image_map.keys()),
        "samples_per_prompt": len(sample_indices),
        "num_prompts": len(prompts),
        "prompts_file": args.prompts_file,
    })

    if "imagereward" in args.metrics:
        evaluate_image_reward(output_root, metric_dir, strategy_to_image_map, sample_indices, prompts, model_dir)
    if "hps" in args.metrics:
        evaluate_hpsv2(output_root, metric_dir, strategy_to_image_map, sample_indices, prompts, args.hps_version)
    if "clip" in args.metrics:
        evaluate_clip(output_root, metric_dir, strategy_to_image_map, sample_indices, prompts, model_dir)

    plot_all(output_root)


if __name__ == "__main__":
    main()
