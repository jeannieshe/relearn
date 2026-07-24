#!/usr/bin/env python
"""Download the Rhaister repo into data/models/Rhaister and install it as an
editable package.

Unlike STATE (scripts/download_ST-HVG-Tahoe.py), Rhaister's Hugging Face repo
ships training code, not a pretrained checkpoint -- there is nothing to run
inference with straight off Hugging Face. This script only does the
"download + install" half; you still need to train it yourself afterwards
(see scripts/README.md) before RhaisterTransitionModel has anything to load.

Usage:
    python scripts/download_Rhaister.py
    cd data/models/Rhaister
    python -m rhaister.train <experiment_name>   # produces the trained artifacts
"""
import subprocess
import sys
from pathlib import Path

# --- model spec -------------------------------------------------------------
REPO_ID = "tahoebio/Rhaister"   # Hugging Face model repo
REVISION = None                 # pin a commit/tag for reproducibility, or None for latest
# destination: <repo root>/data/models/Rhaister (the cfg.env.rhaister_dataset_dir default)
DEST = Path(__file__).resolve().parents[1] / "data" / "models" / "Rhaister"
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required. Install it with:\n    pip install huggingface_hub")

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID}\n  -> {DEST}")
    snapshot_download(repo_id=REPO_ID, revision=REVISION, local_dir=str(DEST))

    if not (DEST / "pyproject.toml").exists():
        sys.exit(
            f"ERROR: expected {DEST / 'pyproject.toml'} missing after download -- "
            "repo layout may have changed."
        )

    print(f"\nInstalling {REPO_ID} as an editable package (uv pip install -e \".[dev]\")...")
    result = subprocess.run(["uv", "pip", "install", "-e", ".[dev]"], cwd=DEST)
    if result.returncode != 0:
        sys.exit(
            "ERROR: `uv pip install -e .[dev]` failed. If `uv` isn't available, retry with:\n"
            f"    pip install -e '{DEST}[dev]'"
        )

    print(
        f"\nDone. {REPO_ID} is downloaded and installed, but NOT yet trained -- there is no\n"
        "pretrained checkpoint on Hugging Face for this model. Train it before\n"
        "RhaisterTransitionModel has anything to load, e.g.:\n\n"
        f"    cd {DEST}\n"
        "    python -m rhaister.train <experiment_name>\n\n"
        "See scripts/README.md for finding/choosing the right experiment name."
    )


if __name__ == "__main__":
    main()
