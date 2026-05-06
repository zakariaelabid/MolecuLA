"""Lightweight checks for the anonymized molecular latent export."""

from __future__ import annotations

import argparse
import csv
import json
import py_compile
import re
import tempfile
from pathlib import Path


EXPORT_ROOT = Path(__file__).resolve().parents[1]


def forbidden_literals() -> list[str]:
    return [
        "C:" + "\\Users",
        "/" + "home" + "/",
        "/" + "data" + "/" + "home2",
        "by" + "outifel",
        "ela" + "bid32",
        "and" + "rze06",
    ]


def iter_text_files(root: Path):
    suffixes = {".py", ".md", ".json", ".csv", ".txt", ".ipynb", ".gitignore", ".gitattributes"}
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix in suffixes or path.name in suffixes):
            yield path


def check_required_assets(skip_checkpoints: bool) -> list[str]:
    required = [
        EXPORT_ROOT / "README.md",
        EXPORT_ROOT / "requirements.txt",
        EXPORT_ROOT / "models/registry.py",
        EXPORT_ROOT / "data/smiles_selfies_full.csv",
        EXPORT_ROOT / "study/results/autoregressive/README.md",
        EXPORT_ROOT / "study/results/autoregressive/extraction_manifest.json",
    ]
    if not skip_checkpoints:
        required.extend(
            [
                EXPORT_ROOT / "checkpoints/linear_attention_h256_l512.pt",
                EXPORT_ROOT / "checkpoints/simple_attention_h256_l256.pt",
            ]
        )
    return [str(path.relative_to(EXPORT_ROOT)) for path in required if not path.exists()]


def check_notebooks_stripped() -> list[str]:
    failures: list[str] = []
    for notebook_path in (EXPORT_ROOT / "study/notebooks").rglob("*.ipynb"):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("execution_count") is not None or cell.get("outputs"):
                failures.append(f"{notebook_path.relative_to(EXPORT_ROOT)} cell {index}")
    return failures


def check_anonymized() -> list[str]:
    failures: list[str] = []
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    for path in iter_text_files(EXPORT_ROOT):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden_literals():
            if token in text:
                failures.append(f"{path.relative_to(EXPORT_ROOT)} contains private path/name token")
        if email_re.search(text):
            failures.append(f"{path.relative_to(EXPORT_ROOT)} contains email-like text")
    return failures


def check_csvs() -> list[str]:
    failures: list[str] = []
    for path in EXPORT_ROOT.rglob("*.csv"):
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                list(csv.reader(handle))
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append(f"{path.relative_to(EXPORT_ROOT)}: {exc}")
    return failures


def check_python_syntax() -> list[str]:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for index, path in enumerate(EXPORT_ROOT.rglob("*.py")):
            try:
                py_compile.compile(str(path), cfile=str(Path(tmpdir) / f"{index}.pyc"), doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{path.relative_to(EXPORT_ROOT)}: {exc.msg}")
    return failures


def check_ar_images() -> list[str]:
    failures: list[str] = []
    root_images = list((EXPORT_ROOT / "study/results/autoregressive/traversals/root_traversals").glob("*.png"))
    latent_images = list((EXPORT_ROOT / "study/results/autoregressive/traversals/latent_traversal_plots").glob("*.png"))
    if len(root_images) != 7:
        failures.append(f"expected 7 root traversal images, found {len(root_images)}")
    if len(latent_images) != 20:
        failures.append(f"expected 20 latent traversal images, found {len(latent_images)}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-checkpoints", action="store_true")
    args = parser.parse_args()

    failures = []
    failures.extend(check_required_assets(args.skip_checkpoints))
    failures.extend(check_notebooks_stripped())
    failures.extend(check_anonymized())
    failures.extend(check_csvs())
    failures.extend(check_python_syntax())
    failures.extend(check_ar_images())

    if failures:
        print("Export verification failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Export verification passed.")


if __name__ == "__main__":
    main()
