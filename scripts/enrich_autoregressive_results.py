"""Rebuild the compact autoregressive result export from source artifacts.

Run this from the source workspace that still has the executed AR notebooks and
legacy traversal image folders:

    python scripts/enrich_autoregressive_results.py --source-root ..

The public export already includes the recovered outputs. This script is kept so
the AR result folder can be regenerated without copying full panels, latent
arrays, probe weights, or private notebook outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path


EXPORT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = EXPORT_ROOT / "study" / "results" / "autoregressive"


def cell_output_texts(notebook_path: Path, cell_index: int) -> list[str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = notebook["cells"][cell_index]
    texts: list[str] = []
    for output in cell.get("outputs", []):
        if "text" in output:
            value = output["text"]
            texts.append("".join(value) if isinstance(value, list) else str(value))
        data = output.get("data") or {}
        plain = data.get("text/plain")
        if plain is not None:
            texts.append("".join(plain) if isinstance(plain, list) else str(plain))
    return [text.replace("\r", "") for text in texts if text]


def parse_prediction_blocks(text: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    blocks = re.split(r"\n=== Predicting ", "\n" + text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        name, _, rest = block.partition(" ===")
        row = {"property": name.strip()}
        confound = re.search(r"confound R2:\s*([-+0-9.eE]+)", rest)
        residual = re.search(r"residual R2:\s*([-+0-9.eE]+)", rest)
        raw = re.search(r"(?m)^R2:\s*([-+0-9.eE]+)", rest)
        if confound:
            row["confound_r2"] = confound.group(1)
        if residual:
            row["residual_r2"] = residual.group(1)
        if raw:
            row["raw_r2"] = raw.group(1)
        rows[row["property"]] = row
    return rows


def merge_probe_blocks(residual_text: str, raw_text: str, source: str, cells: str) -> list[dict[str, str]]:
    residual_rows = parse_prediction_blocks(residual_text)
    raw_rows = parse_prediction_blocks(raw_text)
    properties = list(residual_rows)
    for prop in raw_rows:
        if prop not in residual_rows:
            properties.append(prop)
    rows: list[dict[str, str]] = []
    for prop in properties:
        merged = {
            "property": prop,
            "confound_r2": residual_rows.get(prop, {}).get("confound_r2", ""),
            "residual_r2": residual_rows.get(prop, {}).get("residual_r2", ""),
            "raw_r2": raw_rows.get(prop, {}).get("raw_r2", ""),
            "source_notebook": source,
            "source_cells": cells,
        }
        rows.append(merged)
    return rows


def parse_reconstruction_table(text: str) -> list[dict[str, str]]:
    first: dict[str, dict[str, str]] = {}
    first_pattern = re.compile(
        r"^\s*(\d+)\s+(train|val|test)\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$",
        re.MULTILINE,
    )
    for match in first_pattern.finditer(text):
        first[match.group(1)] = {
            "split": match.group(2),
            "n": match.group(3),
            "token_accuracy": match.group(4),
            "exact_sequence_accuracy": match.group(5),
        }
    second_pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+([-+0-9.eE]+)\s*$", re.MULTILINE)
    for match in second_pattern.finditer(text):
        row = first.get(match.group(1))
        if row:
            row["invalid_decode_count"] = match.group(2)
            row["invalid_decode_rate"] = match.group(3)
    return [row for _, row in sorted(first.items(), key=lambda item: int(item[0]))]


def parse_family_table(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = re.compile(
        r"^\s*\d+\s+([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+(\d+)\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "target_family": match.group(1),
                "valid_fraction": match.group(2),
                "target_family_retention": match.group(3),
                "n_steps": match.group(4),
            }
        )
    return rows


def parse_mlp_raw_table(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = re.compile(
        r"^\s*\d+\s+([A-Za-z0-9\-]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+(\d+)\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "property": match.group(1),
                "r2_train": match.group(2),
                "r2_val": match.group(3),
                "r2_test": match.group(4),
                "epochs": match.group(5),
            }
        )
    return rows


def parse_mlp_residual_table(text: str) -> list[dict[str, str]]:
    full_rows: list[dict[str, str]] = []
    full_pattern = re.compile(
        r"^\s*\d+\s+([A-Za-z0-9\-]+)\s+(resid_[A-Za-z0-9\-]+)\s+"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+(\d+)\s*$",
        re.MULTILINE,
    )
    for match in full_pattern.finditer(text):
        full_rows.append(
            {
                "property": match.group(1),
                "target": match.group(2),
                "r2_train": match.group(3),
                "r2_val": match.group(4),
                "r2_test": match.group(5),
                "epochs": match.group(6),
            }
        )
    if full_rows:
        return full_rows

    first: dict[str, dict[str, str]] = {}
    first_pattern = re.compile(
        r"^\s*(\d+)\s+([A-Za-z0-9\-]+)\s+(resid_[A-Za-z0-9\-]+)\s+"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$",
        re.MULTILINE,
    )
    for match in first_pattern.finditer(text):
        first[match.group(1)] = {
            "property": match.group(2),
            "target": match.group(3),
            "r2_train": match.group(4),
            "r2_val": match.group(5),
        }
    second_pattern = re.compile(r"^\s*(\d+)\s+([-+0-9.eE]+)\s+(\d+)\s*$", re.MULTILINE)
    for match in second_pattern.finditer(text):
        row = first.get(match.group(1))
        if row:
            row["r2_test"] = match.group(2)
            row["epochs"] = match.group(3)
    return [row for _, row in sorted(first.items(), key=lambda item: int(item[0]))]


def parse_overview_table(text: str) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    order: list[str] = []
    first_pattern = re.compile(
        r"^\s*(\d+)\s+([A-Za-z0-9\-]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$",
        re.MULTILINE,
    )
    for match in first_pattern.finditer(text):
        idx = match.group(1)
        order.append(idx)
        rows[idx] = {
            "property": match.group(2),
            "ridge_raw_r2_test": match.group(3),
            "mlp_raw_r2_test": match.group(4),
            "Delta-R2_raw": match.group(5),
        }
    second_pattern = re.compile(
        r"^\s*(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$",
        re.MULTILINE,
    )
    for match in second_pattern.finditer(text):
        idx = match.group(1)
        if idx in rows and "ridge_residual_r2_test" not in rows[idx]:
            rows[idx]["ridge_residual_r2_test"] = match.group(2)
            rows[idx]["mlp_residual_r2_test"] = match.group(3)
            rows[idx]["Delta-R2_residual"] = match.group(4)
    third_pattern = re.compile(r"^\s*(\d+)\s+([-+0-9.eE]+)\s+(.+?)\s*$", re.MULTILINE)
    for match in third_pattern.finditer(text):
        idx = match.group(1)
        interpretation = match.group(3).strip()
        if idx in rows and "abs_spearman" not in rows[idx] and re.search(r"[A-Za-z]", interpretation):
            rows[idx]["abs_spearman"] = match.group(2)
            rows[idx]["interpretation"] = interpretation
    return [rows[idx] for idx in order if idx in rows]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_percent_metric(text: str, name: str) -> float:
    match = re.search(rf"{re.escape(name)}=([0-9.]+)%", text)
    if not match:
        raise ValueError(f"Could not find {name} in AR metric text")
    return float(match.group(1)) / 100.0


def parse_family_macro(text: str) -> float:
    match = re.search(r"Average family retention \(macro\):\s*([-+0-9.eE]+)", text)
    if not match:
        raise ValueError("Could not find family retention macro output")
    return float(match.group(1))


def classify_traversal_image(filename: str, source_group: str) -> tuple[str, str]:
    stem = Path(filename).stem
    if filename == "interproperty_correlation_directional_similarity.png":
        return "all_properties", "interproperty_direction_similarity"
    if stem.endswith("-plot"):
        return stem.removesuffix("-plot"), "property_value_curve"
    plot_type = "root_property_traversal" if source_group == "root_traversals" else "traversal_strip"
    if stem == "clogP":
        return "cLogP", plot_type
    return stem, plot_type


def copy_pngs(source_dir: Path, target_dir: Path, source_group: str, source_label: str) -> list[dict[str, str]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for src in sorted(source_dir.glob("*.png")):
        dst = target_dir / src.name
        shutil.copy2(src, dst)
        prop, plot_type = classify_traversal_image(src.name, source_group)
        rows.append(
            {
                "source_group": source_group,
                "source_folder": source_label,
                "copied_path": dst.relative_to(RESULT_ROOT).as_posix(),
                "filename": src.name,
                "property": prop,
                "plot_type": plot_type,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=EXPORT_ROOT.parent)
    args = parser.parse_args()
    source_root = args.source_root.resolve()

    ar_step1 = source_root / "artifacts/model_compare/ar_model_h256_l256/patched_notebooks/ar_step1_latent_quality.ipynb"
    mlp = source_root / "artifacts/model_compare/ar_model_h256_l256/patched_notebooks/ar_step3_mlp_probes.ipynb"
    mlp_missing = source_root / "artifacts/model_compare/ar_model_h256_l256/patched_notebooks/ar_step3_mlp_missing_property_probes.ipynb"
    train_ar = source_root / "ProbeVAE/train-AR.ipynb"
    latent_analysis = source_root / "ProbeVAE/latent-analisys.ipynb"

    train_metric_text = "\n".join(cell_output_texts(train_ar, 7))
    family_macro_text = "\n".join(cell_output_texts(ar_step1, 14))
    token_accuracy = parse_percent_metric(train_metric_text, "token_acc")
    sequence_accuracy = parse_percent_metric(train_metric_text, "seq_acc")
    family_macro = parse_family_macro(family_macro_text)
    checkpoint_path = EXPORT_ROOT / "checkpoints/H256-L256-3E-2D-Final-NoCorruption.pt"
    checkpoint_available = checkpoint_path.exists() and checkpoint_path.stat().st_size > 1024

    write_csv(
        RESULT_ROOT / "model_validation/metrics.csv",
        [
            {
                "metric": "token_accuracy",
                "value": f"{token_accuracy:.4f}",
                "unit": "fraction",
                "source": "ProbeVAE/train-AR.ipynb cell 7",
            },
            {
                "metric": "sequence_accuracy",
                "value": f"{sequence_accuracy:.4f}",
                "unit": "fraction",
                "source": "ProbeVAE/train-AR.ipynb cell 7",
            },
            {
                "metric": "family_retention_macro",
                "value": str(family_macro),
                "unit": "fraction",
                "source": "artifacts/model_compare/ar_model_h256_l256/patched_notebooks/ar_step1_latent_quality.ipynb cell 14",
            },
            {
                "metric": "checkpoint_available",
                "value": "1" if checkpoint_available else "0",
                "unit": "boolean",
                "source": "export checkpoint inventory",
            },
        ],
        ["metric", "value", "unit", "source"],
    )
    write_json(
        RESULT_ROOT / "model_validation/metrics.json",
        {
            "token_accuracy": token_accuracy,
            "sequence_accuracy": sequence_accuracy,
            "family_retention_macro": family_macro,
            "checkpoint_available": checkpoint_available,
            "checkpoint_path": "checkpoints/H256-L256-3E-2D-Final-NoCorruption.pt",
            "source": {
                "token_and_sequence_accuracy": "ProbeVAE/train-AR.ipynb cell 7",
                "family_retention_macro": "artifacts/model_compare/ar_model_h256_l256/patched_notebooks/ar_step1_latent_quality.ipynb cell 14",
            },
        },
    )

    reconstruction_text = "\n".join(cell_output_texts(ar_step1, 6))
    family_text = "\n".join(cell_output_texts(ar_step1, 12))
    train_resid = "\n".join(cell_output_texts(train_ar, 23))
    train_raw = "\n".join(cell_output_texts(train_ar, 24))
    latent_resid = "\n".join(cell_output_texts(latent_analysis, 15))
    latent_raw = "\n".join(cell_output_texts(latent_analysis, 23))

    write_csv(
        RESULT_ROOT / "model_validation/tables/reconstruction_by_split.csv",
        parse_reconstruction_table(reconstruction_text),
    )
    write_csv(
        RESULT_ROOT / "model_validation/tables/phase5_family_retention_summary.csv",
        parse_family_table(family_text),
    )
    write_csv(
        RESULT_ROOT / "linear_probes/tables/ridge_raw_confound_residual_r2.csv",
        merge_probe_blocks(train_resid, train_raw, "ProbeVAE/train-AR.ipynb", "23,24"),
    )
    write_csv(
        RESULT_ROOT / "linear_probes/tables/linear_regression_raw_confound_residual_r2.csv",
        merge_probe_blocks(latent_resid, latent_raw, "ProbeVAE/latent-analisys.ipynb", "15,23"),
    )

    mlp_texts = cell_output_texts(mlp, 8)
    write_csv(RESULT_ROOT / "mlp_probes/tables/mlp_raw_r2.csv", parse_mlp_raw_table(mlp_texts[2]))
    write_csv(RESULT_ROOT / "mlp_probes/tables/mlp_residual_r2.csv", parse_mlp_residual_table(mlp_texts[3]))
    write_csv(RESULT_ROOT / "mlp_probes/tables/mlp_vs_ridge_overview.csv", parse_overview_table("\n".join(cell_output_texts(mlp, 10))))

    missing_texts = cell_output_texts(mlp_missing, 12)
    write_csv(RESULT_ROOT / "mlp_probes/tables/mlp_missing_raw_r2.csv", parse_mlp_raw_table(missing_texts[1]))
    write_csv(RESULT_ROOT / "mlp_probes/tables/mlp_missing_residual_r2.csv", parse_mlp_residual_table(missing_texts[2]))
    write_csv(
        RESULT_ROOT / "mlp_probes/tables/mlp_missing_vs_ridge_overview.csv",
        parse_overview_table("\n".join(cell_output_texts(mlp_missing, 14))),
    )

    image_rows = []
    image_rows.extend(
        copy_pngs(
            source_root / "Traversals",
            RESULT_ROOT / "traversals/root_traversals",
            "root_traversals",
            "Traversals",
        )
    )
    image_rows.extend(
        copy_pngs(
            source_root / "ProbeVAE/artifacts/latent_traversal_plots",
            RESULT_ROOT / "traversals/latent_traversal_plots",
            "latent_traversal_plots",
            "ProbeVAE/artifacts/latent_traversal_plots",
        )
    )
    write_csv(
        RESULT_ROOT / "traversals/traversal_image_manifest.csv",
        image_rows,
        ["source_group", "source_folder", "copied_path", "filename", "property", "plot_type"],
    )
    write_json(
        RESULT_ROOT / "extraction_manifest.json",
        {
            "model": "autoregressive",
            "status": "recovered_compact_results",
            "checkpoint_status": "included" if checkpoint_available else "missing",
            "notes": [
                "Results were recovered from executed notebook outputs and legacy traversal image folders.",
                "The AR checkpoint is included in the export." if checkpoint_available else "The AR checkpoint is not present in the export.",
                "Large panels, latent arrays, generated probe weights, and full notebook outputs are intentionally excluded.",
            ],
            "source_artifacts": {
                "model_quality": [
                    "artifacts/model_compare/ar_model_h256_l256/patched_notebooks/ar_step1_latent_quality.ipynb",
                    "ProbeVAE/train-AR.ipynb",
                ],
                "linear_probes": ["ProbeVAE/train-AR.ipynb", "ProbeVAE/latent-analisys.ipynb"],
                "mlp_probes": [
                    "artifacts/model_compare/ar_model_h256_l256/patched_notebooks/ar_step3_mlp_probes.ipynb",
                    "artifacts/model_compare/ar_model_h256_l256/patched_notebooks/ar_step3_mlp_missing_property_probes.ipynb",
                ],
                "traversal_images": ["Traversals", "ProbeVAE/artifacts/latent_traversal_plots"],
            },
            "generated_files": {
                "model_validation": [
                    "model_validation/metrics.csv",
                    "model_validation/metrics.json",
                    "model_validation/tables/reconstruction_by_split.csv",
                    "model_validation/tables/phase5_family_retention_summary.csv",
                ],
                "linear_probes": [
                    "linear_probes/tables/ridge_raw_confound_residual_r2.csv",
                    "linear_probes/tables/linear_regression_raw_confound_residual_r2.csv",
                ],
                "mlp_probes": [
                    "mlp_probes/tables/mlp_raw_r2.csv",
                    "mlp_probes/tables/mlp_residual_r2.csv",
                    "mlp_probes/tables/mlp_vs_ridge_overview.csv",
                    "mlp_probes/tables/mlp_missing_raw_r2.csv",
                    "mlp_probes/tables/mlp_missing_residual_r2.csv",
                    "mlp_probes/tables/mlp_missing_vs_ridge_overview.csv",
                ],
                "traversals": [
                    "traversals/traversal_image_manifest.csv",
                    "traversals/root_traversals/*.png",
                    "traversals/latent_traversal_plots/*.png",
                ],
            },
            "image_counts": {
                "root_traversals": sum(row["source_group"] == "root_traversals" for row in image_rows),
                "latent_traversal_plots": sum(row["source_group"] == "latent_traversal_plots" for row in image_rows),
            },
        },
    )


if __name__ == "__main__":
    main()
