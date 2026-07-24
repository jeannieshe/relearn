#!/usr/bin/env python
"""Download the ST-SE-Tahoe STATE model into data/models/ST-SE-Tahoe.

The SE-600M State Embedding counterpart to ST-HVG-Tahoe: this checkpoint was
trained on the 2058-dim X_state embedding rather than the 2000 raw HVGs. Use it
with `env=sw480_se` (see configs/env/sw480_se.yaml).

Fetches only what the relearn agent loads at runtime (see
src/relearn/envs/small_molecules.py):

  fewshot/state_generalization_X_state/checkpoints/best.ckpt   the weights
  fewshot/state_generalization_X_state/pert_onehot_map.pt      the action space

...plus the run's small metadata (config.yaml, var_dims.pkl, the onehot maps,
hparams.yaml) that come with the same HF subfolder. The checkpoint is far too
large to commit to git, so this script pulls it from Hugging Face on demand.

Usage:
    python scripts/download_ST-SE-Tahoe.py
"""
import sys
from pathlib import Path

# --- model spec -------------------------------------------------------------
REPO_ID = "arcinstitute/ST-SE-Tahoe"           # Hugging Face model repo
REVISION = None                                 # pin a commit/tag for reproducibility, or None for latest
RUN = "fewshot/state_generalization_X_state"   # the fewshot run subfolder in the repo
# Grab ONLY what a run needs. Do NOT use "<RUN>/*": that repo also holds
# eval_best.ckpt/ and eval_last.ckpt/ dirs (CSV + multi-GB h5ad eval dumps),
# plus extra final.ckpt/last.ckpt and data_module.torch -- none of which the
# agent loads.
ALLOW_PATTERNS = [
    f"{RUN}/checkpoints/best.ckpt",     # the weights -- required
    f"{RUN}/pert_onehot_map.pt",        # the action space -- required
    # small run metadata: not read on the agent path, but cheap to keep so the
    # bundle also works with `state tx infer --model-dir <RUN>`.
    f"{RUN}/config.yaml",
    f"{RUN}/var_dims.pkl",
    f"{RUN}/batch_onehot_map.pkl",
    f"{RUN}/cell_type_onehot_map.pkl",
    f"{RUN}/version_0/hparams.yaml",
    f"{RUN}/wandb_path.txt",
]
# destination: <repo root>/data/models/ST-SE-Tahoe (the cfg.env.tahoe_dataset_dir
# set by configs/env/sw480_se.yaml)
DEST = Path(__file__).resolve().parents[1] / "data" / "models" / "ST-SE-Tahoe"
# files the agent actually opens, relative to DEST -- checked after download
REQUIRED = [
    f"{RUN}/checkpoints/best.ckpt",
    f"{RUN}/pert_onehot_map.pt",
]
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required. Install it with:\n    pip install huggingface_hub")

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} (patterns={ALLOW_PATTERNS})\n  -> {DEST}")
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(DEST),
    )

    missing = [f for f in REQUIRED if not (DEST / f).exists()]
    if missing:
        sys.exit("ERROR: expected files missing after download:\n  " + "\n  ".join(missing))

    print("\nRequired model files present:")
    for f in REQUIRED:
        size_mb = (DEST / f).stat().st_size / 1e6
        print(f"  {size_mb:8.1f} MB  {f}")

    # Neutral-state cache: NOT on Hugging Face. It is the mean DMSO X_state profile
    # for the configured cell line, auto-generated on the first env init from
    # cfg.tahoe_se_dir (requires cluster access). Present here only if generated.
    npy = DEST / "SW480_dmso_neutral_state.npy"
    if npy.exists():
        print(f"  {npy.stat().st_size / 1e3:8.1f} KB  SW480_dmso_neutral_state.npy (cached)")
    else:
        print(
            "\nNote: SW480_dmso_neutral_state.npy is not present. It is auto-generated on the\n"
            "first env init from cfg.tahoe_se_dir (needs cluster access); the first run is\n"
            "slower because it reads the multi-GB Tahoe-SE h5ad, then caches the small result."
        )

    print(f"\nDone. Run with:\n  python src/relearn/agents/dqn.py env=sw480_se")


if __name__ == "__main__":
    main()
