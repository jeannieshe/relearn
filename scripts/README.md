# scripts/

Utility scripts for setting up a checkout so the code runs without touching the
notebooks. Large model artifacts (checkpoints, pert maps) are **not** committed
to git -- they live on Hugging Face and are fetched on demand.

## Model downloaders

One script per model, named `download_<MODEL>.py`, so you run only what you need:

| Script | Model | Downloads into |
|---|---|---|
| `download_ST-HVG-Tahoe.py` | `arcinstitute/ST-HVG-Tahoe` (2000-HVG STATE, `state_generalization_X_hvg` fewshot run) | `data/models/ST-HVG-Tahoe/` |
| `download_ST-SE-Tahoe.py` | `arcinstitute/ST-SE-Tahoe` (2058-dim SE-600M embedding, `state_generalization_X_state` fewshot run) | `data/models/ST-SE-Tahoe/` |

Run from the repo root:

```bash
python scripts/download_ST-HVG-Tahoe.py   # for env=sw480 (default, raw HVGs)
python scripts/download_ST-SE-Tahoe.py    # for env=sw480_se (SE-600M embedding)
```

This pulls the checkpoint and the pert-onehot map (plus small run metadata) into
`data/models/<MODEL>/`, which is the `cfg.env.tahoe_dataset_dir` for that env.
Pick the downloader matching the `env=` you run: `sw480` needs ST-HVG-Tahoe,
`sw480_se` needs ST-SE-Tahoe.

### Adding a new model

1. Copy `download_ST-HVG-Tahoe.py` to `download_<MODEL>.py`.
2. Edit the `model spec` block: `REPO_ID`, `ALLOW_PATTERNS`, `DEST`, `REQUIRED`.
3. Add a row to the table above.

## Notes

- `SW480_dmso_neutral_<hvg|state>.npy` (the neutral start-state cache, one per
  embedding) is not on Hugging Face; it is auto-generated on the first env init
  from `cfg.tahoe_se_dir`, which requires cluster access. The download script
  reports whether it is present.
- These artifacts are covered by `.gitignore` (`data/`, `*.ckpt`, `*.pt`, ...) on
  purpose -- re-download rather than commit.
