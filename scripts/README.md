# scripts/

Utility scripts for setting up a checkout so the code runs without touching the
notebooks. Large model artifacts (checkpoints, pert maps) are **not** committed
to git -- they live on Hugging Face and are fetched on demand.

## Model downloaders

One script per model, named `download_<MODEL>.py`, so you run only what you need:

| Script | Model | Downloads into |
|---|---|---|
| `download_ST-HVG-Tahoe.py` | `arcinstitute/ST-HVG-Tahoe` (2000-HVG STATE, `state_generalization_X_hvg` fewshot run) | `data/models/ST-HVG-Tahoe/` |

Run from the repo root:

```bash
python scripts/download_ST-HVG-Tahoe.py
```

This pulls the ~1 GB checkpoint and the pert-onehot map (plus small run metadata)
into `data/models/ST-HVG-Tahoe/`, which is the default `cfg.env.tahoe_dataset_dir`.

### Adding a new model

1. Copy `download_ST-HVG-Tahoe.py` to `download_<MODEL>.py`.
2. Edit the `model spec` block: `REPO_ID`, `ALLOW_PATTERNS`, `DEST`, `REQUIRED`.
3. Add a row to the table above.

## Notes

- `SW480_dmso_neutral_hvg.npy` (the neutral start-state cache) is not on Hugging
  Face; it is auto-generated on the first env init from `cfg.tahoe_se_dir`, which
  requires cluster access. The download script reports whether it is present.
- These artifacts are covered by `.gitignore` (`data/`, `*.ckpt`, `*.pt`, ...) on
  purpose -- re-download rather than commit.
