from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.severity_pairs import LEVEL, PAIR_NAMES, PairRecord, TYPES


def ordinal_targets(levels: torch.Tensor) -> torch.Tensor:
    thresholds = torch.arange(3, device=levels.device)[None, :]
    return (levels[:, None] > thresholds).float()


@torch.no_grad()
def predict_image_dataset(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str = "bf16",
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    encoder.eval()
    use_amp = device.type == "cuda" and precision.lower() in {"bf16", "bfloat16", "fp16", "float16"}
    amp_dtype = torch.bfloat16 if precision.lower() in {"bf16", "bfloat16"} else torch.float16
    names: List[str] = []
    severity_chunks: List[np.ndarray] = []
    probability_chunks: List[np.ndarray] = []
    class_chunks: List[np.ndarray] = []
    for batch in tqdm(loader, desc="Severity inference"):
        image = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits, severity, probabilities, class_logits, _ = encoder.forward_with_aux(image)
        names.extend([str(value) for value in batch["name"]])
        severity_chunks.append(severity.float().cpu().numpy())
        probability_chunks.append(probabilities.float().cpu().numpy())
        class_chunks.append(class_logits.float().cpu().numpy())
    return (
        names,
        np.concatenate(severity_chunks, axis=0).reshape(-1),
        np.concatenate(probability_chunks, axis=0),
        np.concatenate(class_chunks, axis=0),
    )


def evaluate_severity_predictions(
    names: Sequence[str],
    severity: np.ndarray,
    probabilities: np.ndarray,
    class_logits: np.ndarray,
    records: Sequence[PairRecord],
    anchor_targets: Mapping[str, float],
    margin_base: float = 0.08,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    prediction = {name: float(value) for name, value in zip(names, severity)}

    pair_rows: List[Dict[str, object]] = []
    for record in records:
        low_values = [prediction[name] for name in record.low_files if name in prediction]
        high_values = [prediction[name] for name in record.high_files if name in prediction]
        if not low_values or not high_values:
            continue
        low_score = float(np.mean(low_values))
        high_score = float(np.mean(high_values))
        required_margin = float(margin_base * record.level_gap)
        pair_rows.append(
            {
                "group_id": record.group_id,
                "pair": record.pair_name,
                "low_type": record.low_type,
                "high_type": record.high_type,
                "low_score": low_score,
                "high_score": high_score,
                "score_gap": high_score - low_score,
                "required_margin": required_margin,
                "ordered": high_score > low_score,
                "margin_ordered": high_score >= low_score + required_margin,
            }
        )
    pair_df = pd.DataFrame(pair_rows)

    sample_rows: List[Dict[str, object]] = []
    class_prediction = np.argmax(class_logits, axis=1)
    for idx, name in enumerate(names):
        sample_type = name.split("_", 1)[0].lower()
        if sample_type not in LEVEL:
            continue
        level = LEVEL[sample_type]
        target = float(anchor_targets[sample_type])
        predicted_level = int((probabilities[idx] >= 0.5).sum())
        sample_rows.append(
            {
                "name": name,
                "type": sample_type,
                "level": level,
                "severity": float(severity[idx]),
                "target": target,
                "abs_error": abs(float(severity[idx]) - target),
                "ordinal_level": predicted_level,
                "ordinal_correct": predicted_level == level,
                "class_level": int(class_prediction[idx]),
                "class_correct": int(class_prediction[idx]) == level,
                "p1": float(probabilities[idx, 0]),
                "p2": float(probabilities[idx, 1]),
                "p3": float(probabilities[idx, 2]),
            }
        )
    sample_df = pd.DataFrame(sample_rows)

    if len(pair_df):
        overall_pair_accuracy = float(pair_df["ordered"].mean())
        degraded_pairs = pair_df[~pair_df["pair"].str.startswith("identity<")]
        degraded_pair_accuracy = (
            float(degraded_pairs["ordered"].mean()) if len(degraded_pairs) else None
        )
        margin_pair_accuracy = float(pair_df["margin_ordered"].mean())
        per_pair_df = (
            pair_df.groupby("pair", sort=False)
            .agg(
                n=("pair", "size"),
                accuracy=("ordered", "mean"),
                margin_accuracy=("margin_ordered", "mean"),
                mean_gap=("score_gap", "mean"),
                median_gap=("score_gap", "median"),
            )
            .reindex(PAIR_NAMES)
        )
        per_pair = {
            pair_name: {
                key: (None if pd.isna(value) else float(value))
                for key, value in row.items()
            }
            for pair_name, row in per_pair_df.to_dict(orient="index").items()
        }
    else:
        overall_pair_accuracy = None
        degraded_pair_accuracy = None
        margin_pair_accuracy = None
        per_pair = {}

    type_table = (
        sample_df.groupby("type")
        .agg(
            n=("name", "size"),
            severity_mean=("severity", "mean"),
            severity_std=("severity", "std"),
            severity_min=("severity", "min"),
            severity_max=("severity", "max"),
            mae=("abs_error", "mean"),
            ordinal_accuracy=("ordinal_correct", "mean"),
            class_accuracy=("class_correct", "mean"),
        )
        .reindex(TYPES)
    )
    type_means = {
        sample_type: (
            None
            if pd.isna(type_table.loc[sample_type, "severity_mean"])
            else float(type_table.loc[sample_type, "severity_mean"])
        )
        for sample_type in TYPES
    }

    all_spearman = spearmanr(sample_df["level"], sample_df["severity"])
    degraded_samples = sample_df[sample_df["type"] != "identity"]
    degraded_spearman = spearmanr(
        degraded_samples["level"], degraded_samples["severity"]
    )
    adjacent_order = all(
        type_means[TYPES[idx]] is not None
        and type_means[TYPES[idx + 1]] is not None
        and type_means[TYPES[idx]] < type_means[TYPES[idx + 1]]
        for idx in range(3)
    )

    mild_medium_accuracy = (per_pair.get("mild<medium") or {}).get("accuracy")
    medium_strong_accuracy = (per_pair.get("medium<strong") or {}).get("accuracy")
    summary: Dict[str, object] = {
        "overall_pair_accuracy": overall_pair_accuracy,
        "degraded_pair_accuracy": degraded_pair_accuracy,
        "margin_pair_accuracy": margin_pair_accuracy,
        "per_pair": per_pair,
        "num_images": int(len(sample_df)),
        "severity_mae": float(sample_df["abs_error"].mean()),
        "ordinal_level_accuracy": float(sample_df["ordinal_correct"].mean()),
        "class_level_accuracy": float(sample_df["class_correct"].mean()),
        "spearman_all": float(all_spearman.statistic),
        "spearman_all_p": float(all_spearman.pvalue),
        "spearman_degraded": float(degraded_spearman.statistic),
        "spearman_degraded_p": float(degraded_spearman.pvalue),
        "type_means": type_means,
        "strict_type_mean_order": bool(adjacent_order),
        "type_table": {
            sample_type: {
                key: (None if pd.isna(value) else float(value))
                for key, value in row.items()
            }
            for sample_type, row in type_table.to_dict(orient="index").items()
        },
        "passes_validation": bool(
            adjacent_order
            and overall_pair_accuracy is not None
            and overall_pair_accuracy >= 0.95
            and degraded_pair_accuracy is not None
            and degraded_pair_accuracy >= 0.90
            and mild_medium_accuracy is not None
            and mild_medium_accuracy >= 0.90
            and medium_strong_accuracy is not None
            and medium_strong_accuracy >= 0.85
            and float(degraded_spearman.statistic) >= 0.70
            and float(sample_df["abs_error"].mean()) <= 0.15
        ),
    }
    return summary, pair_df, sample_df


def severity_validation_score(summary: Mapping[str, object]) -> Tuple[float, ...]:
    per_pair = summary.get("per_pair", {})
    if not isinstance(per_pair, Mapping):
        per_pair = {}
    mild_medium = (per_pair.get("mild<medium", {}) or {}).get("accuracy", 0.0)
    medium_strong = (per_pair.get("medium<strong", {}) or {}).get("accuracy", 0.0)
    return (
        float(bool(summary.get("passes_validation", False))),
        float(mild_medium or 0.0),
        float(summary.get("degraded_pair_accuracy") or 0.0),
        float(summary.get("overall_pair_accuracy") or 0.0),
        float(medium_strong or 0.0),
        float(summary.get("spearman_degraded") or -1.0),
        -float(summary.get("severity_mae") or 999.0),
        float(summary.get("class_level_accuracy") or 0.0),
    )


def save_severity_report(
    output_dir: str | Path,
    summary: Dict[str, object],
    pair_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(output_dir / "pair_metrics.csv", index=False, encoding="utf-8-sig")
    sample_df.to_csv(
        output_dir / "sample_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
