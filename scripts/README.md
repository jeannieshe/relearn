# scripts/

Utility scripts for setting up a checkout so the code runs without touching the
notebooks. Large model artifacts (checkpoints, pert maps) are **not** committed
to git -- they live on Hugging Face and are fetched on demand.

## Model downloaders

One script per model, named `download_<MODEL>.py`, so you run only what you need:

| Script | Model | Downloads into |
|---|---|---|
| `download_ST-HVG-Tahoe.py` | `arcinstitute/ST-HVG-Tahoe` (2000-HVG STATE, `state_generalization_X_hvg` fewshot run) | `data/models/ST-HVG-Tahoe/` |
<<<<<<< Updated upstream
| `download_ST-SE-Tahoe.py` | `arcinstitute/ST-SE-Tahoe` (2058-dim SE-600M embedding, `state_generalization_X_state` fewshot run) | `data/models/ST-SE-Tahoe/` |
=======
| `download_Rhaister.py` | `tahoebio/Rhaister` (additive-ALS + ridge + calibration MLP) | `data/models/Rhaister/` |
>>>>>>> Stashed changes

Run from the repo root:

```bash
python scripts/download_ST-HVG-Tahoe.py   # for env=sw480 (default, raw HVGs)
python scripts/download_ST-SE-Tahoe.py    # for env=sw480_se (SE-600M embedding)
```

This pulls the checkpoint and the pert-onehot map (plus small run metadata) into
`data/models/<MODEL>/`, which is the `cfg.env.tahoe_dataset_dir` for that env.
Pick the downloader matching the `env=` you run: `sw480` needs ST-HVG-Tahoe,
`sw480_se` needs ST-SE-Tahoe.

### Rhaister: download + install is not enough -- you must also train it

Unlike STATE, Rhaister's Hugging Face repo ships training code, not a
pretrained checkpoint -- it "trains in seconds, predicts in milliseconds," so
Tahoe Bio distributes it as something you train yourself rather than a
checkpoint to download. `download_Rhaister.py` only does the fetch-and-install
half (`snapshot_download` + `uv pip install -e ".[dev]"`); after that, train
it explicitly:

```bash
python scripts/download_Rhaister.py
cd data/models/Rhaister
python -m rhaister.train <experiment_name>
```

Check `data/models/Rhaister/experiments/` (or equivalent, per the downloaded
repo's own layout) for an existing Tahoe-100M fewshot experiment config before
writing a new one -- Tahoe-100M is one of Rhaister's four training datasets,
so one likely already exists. `cfg.env.rhaister_experiment_name`
(`src/relearn/config.py`) must match whatever experiment name you train.

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
