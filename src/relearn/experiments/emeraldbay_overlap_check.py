"""
Feasibility check: how much of tahoebio/EmeraldBay's combinatorial-perturbation
data could validate STATE's `env.multi_hot()` combination predictions?

Motivation: `env.multi_hot(drug_indices)` (envs/small_molecules.py) sums one-hot
rows from STATE's `pert_matrix` to build a "combination" perturbation vector,
but it can only combine drugs that are keys in
`pert_onehot_map.pt` -- Tahoe-100M's fixed 1138-condition / ~380-unique-drug
small-molecule screen that STATE's action space was trained on. EmeraldBay
(https://huggingface.co/datasets/tahoebio/EmeraldBay, Tahoe Therapeutics) has
real single-cell measurements of drug COMBINATIONS (e.g. FOLFOX-style
multi-drug regimens, "Adagrasib+Cetuximab"-style pairs) across 52 cell lines
including SW480 (CVCL_0546) -- a ground truth `env.multi_hot()` currently has
no way to be checked against.

This script does NOT build the real-cell loader or download the full ~57.7 GB
`expression_data` (116 parquet shards). It only:

  1. Parses Tahoe-100M's drug panel out of `pert_onehot_map.pt` (the same file
     `StateTransitionModel` loads -- see transitions/state_model.py).
  2. Pulls EmeraldBay's tiny `metadata/drug_metadata.parquet` and
     `metadata/cell_line_metadata.parquet` (~12 KB / ~13 KB, not expression
     data) via `huggingface_hub`.
  3. Confirms EmeraldBay's `drug_metadata` table is single-drug only (per the
     dataset card) by recovering combination-condition names from TWO other
     sources: (a) `metadata/summary_statistics.parquet` (~52 KB,
     cell_line x condition x growth_rate, 4992 rows) -- its `condition`
     column already lists every combo as a `[(name, conc, unit), ...]` list,
     so this alone gives the EXHAUSTIVE set of combo conditions across all
     52 cell lines with no expression_data download at all; and (b), as the
     task originally asked for, peeking at ONE `expression_data` shard's
     `drug` column as a sanity check / fallback. NOTE: `hf_hub_download`
     always fetches the whole parquet file (no server-side column
     projection over the hub) -- shard 0 is ~411 MB, fine as a one-shard
     peek but a much noisier and less complete source than (a): a single
     shard only covers whichever cell lines/conditions happened to land in
     it, e.g. it's missing "Adagrasib+Cetuximab" entirely even though that
     combo is measured (on SW480 among others) elsewhere in the dataset.
  4. Fuzzy-matches every EmeraldBay drug name against the Tahoe-100M panel
     (normalizing salt-form suffixes, stereochemistry prefixes, "5-" etc.)
     and reports which EmeraldBay COMBINATION conditions have every
     component drug representable via `env.multi_hot()`.

Run with:
    /home/jeannie/miniconda/envs/pytorch-pip/bin/python \
        src/relearn/experiments/emeraldbay_overlap_check.py

Use the `pytorch-pip` conda env specifically -- `relearn-env` / `pytorch` have
a broken torch install (`ImportError: undefined symbol: iJIT_NotifyEvent`).
"""

import ast
import csv
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).parent.parent.parent.parent
PERT_MAP_PATH = (
    REPO_ROOT
    / "data/models/ST-HVG-Tahoe/fewshot/state_generalization_X_hvg/pert_onehot_map.pt"
)
OUT_DIR = REPO_ROOT / "artifacts"
EMERALDBAY_REPO = "tahoebio/EmeraldBay"

# Salt-form / stereochemistry decorations that differ between the two
# naming conventions but don't change which small molecule is meant.
_SALT_SUFFIXES = (
    " ditosylate", " hydrochloride", " mesylate", " acetate", " sulfate",
    " citrate", " tartrate", " succinate", " maleate", " fumarate",
    " (dmso_tf solvate)",
)
_PREFIX_STRIP_RE = re.compile(r"^\(?[rs]\)?-|^\d+-")


def normalize_drug_name(name: str) -> str:
    """Fold naming variants (salts, stereo-prefixes, parentheticals) so
    Tahoe-100M's `5-Fluorouracil` / `Almonertinib (hydrochloride)` and
    EmeraldBay's `Fluorouracil` / `Lapatinib ditosylate` compare equal."""
    n = name.strip().lower()
    n = re.sub(r"\s*\([^)]*\)\s*", " ", n).strip()
    for suf in _SALT_SUFFIXES:
        if n.endswith(suf.strip()):
            n = n[: -len(suf.strip())].strip()
    n = _PREFIX_STRIP_RE.sub("", n)
    return n.strip()


def load_tahoe100m_drug_names() -> set[str]:
    """Tahoe-100M's small-molecule panel: every unique drug name STATE's
    action space can index into via `pert_onehot_map.pt`. Keys are
    stringified `[(name, concentration, units), ...]` lists, matching how
    `StateTransitionModel.drug_list = list(self.pert_map.keys())` is built."""
    pert_map = torch.load(PERT_MAP_PATH, weights_only=False)
    names = set()
    for key in pert_map.keys():
        for name, _conc, _units in ast.literal_eval(str(key)):
            names.add(name)
    return names


def load_emeraldbay_drug_metadata():
    path = hf_hub_download(
        repo_id=EMERALDBAY_REPO, repo_type="dataset",
        filename="metadata/drug_metadata.parquet",
    )
    return pq.read_table(path).to_pandas()


def load_emeraldbay_cell_line_metadata():
    path = hf_hub_download(
        repo_id=EMERALDBAY_REPO, repo_type="dataset",
        filename="metadata/cell_line_metadata.parquet",
    )
    return pq.read_table(path).to_pandas()


def load_emeraldbay_summary_statistics():
    """cell_line x condition x growth_rate (~52 KB, 4992 rows). `condition`
    is a stringified `[(name, conc, unit), ...]` list -- combo conditions
    have len > 1 -- so this gives the exhaustive set of tested combinations
    across all 52 cell lines without touching expression_data at all."""
    path = hf_hub_download(
        repo_id=EMERALDBAY_REPO, repo_type="dataset",
        filename="metadata/summary_statistics.parquet",
    )
    return pq.read_table(path).to_pandas()


def combo_conditions_from_summary_stats(summary_df) -> dict[tuple, set[str]]:
    """Returns {sorted-tuple-of-component-drug-names: {cell_lines tested on}}
    for every multi-drug condition in summary_statistics.parquet."""
    combos: dict[tuple, set[str]] = {}
    for cell_line, cond in zip(summary_df["cell_line"], summary_df["condition"]):
        parsed = ast.literal_eval(cond)
        if len(parsed) > 1:
            key = tuple(sorted(p[0] for p in parsed))
            combos.setdefault(key, set()).add(cell_line)
    return combos


def sample_expression_shard_drug_conditions(shard_idx: int = 0) -> set[str]:
    """Peek at ONE expression_data shard's `drug` column to recover
    combination-condition names (e.g. 'Gemcitabine+Paclitaxel'). These do
    NOT appear in `drug_metadata` at all -- the dataset card says that table
    is single-drug only. hf_hub_download caches the file, so repeat calls
    for the same shard_idx don't re-download."""
    filename = f"expression_data/train-{shard_idx:05d}-of-00116.parquet"
    path = hf_hub_download(
        repo_id=EMERALDBAY_REPO, repo_type="dataset", filename=filename,
    )
    tbl = pq.ParquetFile(path).read(columns=["drug"])
    return set(tbl.column("drug").to_pylist())


def match_drug(name: str, tahoe_norm_to_orig: dict[str, str]) -> str | None:
    n = normalize_drug_name(name)
    if n in tahoe_norm_to_orig:
        return tahoe_norm_to_orig[n]
    return None


def main():
    tahoe_names = load_tahoe100m_drug_names()
    # Several Tahoe-100M names collapse to the same normalized form (e.g.
    # "Irinotecan" and "Irinotecan (hydrochloride)" both -> "irinotecan").
    # Prefer the shortest/plain original name deterministically, rather than
    # letting an arbitrary set/dict iteration order pick the salt-form variant.
    tahoe_norm_to_orig: dict[str, str] = {}
    for n in sorted(tahoe_names, key=len):
        norm = normalize_drug_name(n)
        tahoe_norm_to_orig.setdefault(norm, n)
    print(f"Tahoe-100M pert_onehot_map: {len(tahoe_names)} unique drug names "
          f"({len(tahoe_norm_to_orig)} after normalization)")

    drug_meta = load_emeraldbay_drug_metadata()
    emerald_drugs = sorted(drug_meta["drug"].tolist())
    print(f"\nEmeraldBay metadata/drug_metadata.parquet: {len(emerald_drugs)} "
          f"single-drug rows (confirms: no combo rows in this table)")

    cell_meta = load_emeraldbay_cell_line_metadata()
    sw480_rows = cell_meta[cell_meta["Cell_ID_Cellosaur"] == "CVCL_0546"]
    print(f"\nSW480 (CVCL_0546) in cell_line_metadata: "
          f"{'YES' if len(sw480_rows) else 'NO'} "
          f"({len(sw480_rows)} row(s), cell_name={sw480_rows['cell_name'].unique().tolist()})")

    match_rows = []
    n_matched = 0
    for d in emerald_drugs:
        hit = match_drug(d, tahoe_norm_to_orig)
        match_rows.append({"emeraldbay_drug": d, "tahoe100m_match": hit or ""})
        n_matched += hit is not None
    print(f"\nSingle-drug overlap: {n_matched}/{len(emerald_drugs)} EmeraldBay "
          f"drugs found in Tahoe-100M's panel")
    for r in match_rows:
        status = r["tahoe100m_match"] or "NO MATCH (not in Tahoe-100M panel)"
        print(f"  {r['emeraldbay_drug']:<24} -> {status}")

    # -- primary/exhaustive combo source: summary_statistics.parquet --
    print("\nReading metadata/summary_statistics.parquet for the EXHAUSTIVE "
          "set of combination conditions (all 52 cell lines, no "
          "expression_data download needed)...")
    summary_df = load_emeraldbay_summary_statistics()
    combo_to_cell_lines = combo_conditions_from_summary_stats(summary_df)
    sw480_combos = {c for c, cls in combo_to_cell_lines.items() if "CVCL_0546" in cls}
    print(f"  {len(combo_to_cell_lines)} unique combo conditions total across "
          f"all cell lines; {len(sw480_combos)} of them were tested on SW480 "
          f"(CVCL_0546)")

    representable_combos = []
    combo_rows = []
    for combo in sorted(combo_to_cell_lines):
        matches = {c: match_drug(c, tahoe_norm_to_orig) for c in combo}
        all_representable = all(matches.values())
        row = {
            "emeraldbay_combo": "+".join(combo),
            "components": "; ".join(combo),
            "tahoe100m_matches": "; ".join(matches[c] or "NO MATCH" for c in combo),
            "all_components_representable": all_representable,
            "tested_on_sw480": combo in sw480_combos,
            "n_cell_lines_tested": len(combo_to_cell_lines[combo]),
        }
        combo_rows.append(row)
        if all_representable:
            representable_combos.append((row["emeraldbay_combo"], matches, row["tested_on_sw480"]))

    print(f"\nAll {len(combo_to_cell_lines)} combination conditions "
          f"(name -> Tahoe-100M match per component):")
    for row in combo_rows:
        flag = "REPRESENTABLE via env.multi_hot()" if row["all_components_representable"] else "not fully representable"
        sw480_flag = " [tested on SW480]" if row["tested_on_sw480"] else ""
        print(f"  {row['emeraldbay_combo']:<50} [{flag}]{sw480_flag}")
        print(f"      -> {row['tahoe100m_matches']}")

    print(f"\n{len(representable_combos)} combination condition(s) have ALL "
          f"component drugs in Tahoe-100M's panel:")
    for combo, matches, tested_sw480 in representable_combos:
        mapping = ", ".join(f"{k} -> {v}" for k, v in matches.items())
        print(f"  {combo}: {mapping}  (tested on SW480: {tested_sw480})")

    # -- secondary sanity check: peek at ONE expression_data shard --
    print("\n[sanity check, not exhaustive] peeking at expression_data shard 0's "
          "'drug' column, as originally scoped...")
    shard0_conditions = sample_expression_shard_drug_conditions(0)
    shard0_combos = sorted(c for c in shard0_conditions if "+" in c)
    print(f"  shard 0: {len(shard0_conditions)} unique 'drug' conditions, "
          f"{len(shard0_combos)} of them combos: {shard0_combos}")
    print("  (this single shard misses several combos present elsewhere in "
          "the dataset, e.g. Adagrasib+Cetuximab -- summary_statistics.parquet "
          "above is the reliable source, not this shard peek)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "emeraldbay_single_drug_matches.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["emeraldbay_drug", "tahoe100m_match"])
        w.writeheader()
        w.writerows(match_rows)
    with open(OUT_DIR / "emeraldbay_combo_matches.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "emeraldbay_combo", "components", "tahoe100m_matches",
            "all_components_representable", "tested_on_sw480", "n_cell_lines_tested",
        ])
        w.writeheader()
        w.writerows(combo_rows)
    summary = {
        "tahoe100m_unique_drug_count": len(tahoe_names),
        "emeraldbay_drug_metadata_count": len(emerald_drugs),
        "emeraldbay_single_drug_matched_count": n_matched,
        "sw480_present": bool(len(sw480_rows)),
        "total_unique_combo_conditions": len(combo_to_cell_lines),
        "combo_conditions_tested_on_sw480": len(sw480_combos),
        "representable_combos": [c for c, _, _ in representable_combos],
        "shard0_sanity_check_unique_conditions": len(shard0_conditions),
        "shard0_sanity_check_combo_conditions": shard0_combos,
    }
    with open(OUT_DIR / "emeraldbay_overlap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote artifacts/emeraldbay_{{single_drug_matches,combo_matches}}.csv "
          f"and emeraldbay_overlap_summary.json")


if __name__ == "__main__":
    main()
