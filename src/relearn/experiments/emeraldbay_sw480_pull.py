"""
Targeted pull of SW480 (CVCL_0546) rows from tahoebio/EmeraldBay's
`expression_data` (116 parquet shards, ~57.7 GB total) -- NOT the full
densification/normalization pipeline (that's a later phase).

Prior phase (emeraldbay_overlap_check.py, artifacts/emeraldbay_overlap_summary.json)
found exactly 2 of EmeraldBay's 16 combo conditions have every component drug in
STATE's Tahoe-100M panel AND are tested on SW480: "Dabrafenib+Trametinib" and
"Gemcitabine+Paclitaxel". This script pulls, for cell_line == CVCL_0546 only:
  (a) DMSO controls
  (b) Gemcitabine+Paclitaxel-treated cells
  (c) Dabrafenib+Trametinib-treated cells

Method (chosen after two things were ruled out empirically, see report):
  1. HF datasets-server `/filter` endpoint: supported (`"filter": true` in
     `/is-valid`), but its index only covers the first 5 GB of a >5GB dataset
     ("partial": true was returned), i.e. roughly the first ~11 of 116 shards.
     A `/rows` probe across the full row range showed CVCL_0546 rows are NOT
     concentrated in early shards -- they appear at roughly similar density
     (~10-30% per 10-row window) all the way to offset 1.77M of 1.83M total
     rows. So `/filter` alone would silently miss most matches. Not used.
  2. `datasets.load_dataset(..., streaming=True)`: the documented fallback,
     but reads full rows (including the ~2000-dim genes+expressions arrays,
     the vast majority of each row's bytes) for every row it iterates past,
     even non-matches.
  3. What this script actually does: `huggingface_hub.HfFileSystem` (fsspec,
     supports HTTP range requests) + `pyarrow.parquet.ParquetFile`, opened
     per shard WITHOUT downloading the whole file:
       pass 1 (cheap): read only the `cell_line`/`drug`/`drugname_drugconc`
       columns for the whole shard (~4-6 sec/shard measured on shard 0,
       vs. a ~400-700MB full download) to find which row groups contain any
       matching row.
       pass 2 (targeted): `ParquetFile.read_row_group(i)` (all columns) only
       for row groups that pass 1 flagged -- still a range-request read, not
       a full-file download -- then filter to the exact matching rows.
  This keeps total transfer to roughly (116 shards x cheap metadata columns)
  + (full bytes of only the row groups that actually contain a SW480 match),
  far below 57.7 GB.

Run with:
    /home/jeannie/miniconda/envs/pytorch-pip/bin/python \
        src/relearn/experiments/emeraldbay_sw480_pull.py
"""

import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "data/datasets/EmeraldBay/SW480"
EMERALDBAY_REPO = "tahoebio/EmeraldBay"
N_SHARDS = 116
CELL_LINE = "CVCL_0546"

# Both drug-name orderings observed in the raw `drug` column (e.g. shard 0
# sanity check in emeraldbay_overlap_check.py found both
# "Gemcitabine+Paclitaxel" and "Paclitaxel+Gemcitabine" as distinct strings).
COMBO_SETS = {
    "gemcitabine_paclitaxel": frozenset({"Gemcitabine", "Paclitaxel"}),
    "dabrafenib_trametinib": frozenset({"Dabrafenib", "Trametinib"}),
}


def drug_set(drug_str: str) -> frozenset[str]:
    return frozenset(p.strip() for p in drug_str.split("+"))


def is_dmso(drug_str: str) -> bool:
    # DMSO_T0 (time-zero baseline, pre-culture) is NOT the same condition as
    # DMSO_TF (terminal vehicle control, harvested at the same 5-day endpoint
    # as the drug-treated cells) -- mixing them in would bias any control-vs-
    # treated comparison. Match DMSO_TF only, mirroring this repo's own
    # convention (EnvConfig.dmso_control_pert = "[('DMSO_TF', 0.0, 'uM')]").
    return drug_str.strip().upper() == "DMSO_TF"


def classify(drug_str: str) -> str | None:
    if is_dmso(drug_str):
        return "dmso_control"
    ds = drug_set(drug_str)
    for combo_name, combo_set in COMBO_SETS.items():
        if ds == combo_set:
            return combo_name
    return None


def shard_path(idx: int) -> str:
    return (
        f"datasets/{EMERALDBAY_REPO}/expression_data/"
        f"train-{idx:05d}-of-{N_SHARDS:05d}.parquet"
    )


def main():
    fs = HfFileSystem()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[pa.Table]] = {
        "dmso_control": [],
        "gemcitabine_paclitaxel": [],
        "dabrafenib_trametinib": [],
    }
    seen_dmso_labels: set[str] = set()
    seen_combo_drugname_drugconc: dict[str, set[str]] = {
        "gemcitabine_paclitaxel": set(),
        "dabrafenib_trametinib": set(),
    }

    t_start = time.time()
    total_rows_scanned = 0
    total_row_groups_fully_read = 0

    for shard_idx in range(N_SHARDS):
        path = shard_path(shard_idx)
        t0 = time.time()
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            n_row_groups = pf.num_row_groups
            # cheap pass: only the 3 small string columns, whole shard
            meta_tbl = pf.read(columns=["cell_line", "drug", "drugname_drugconc"])
            total_rows_scanned += meta_tbl.num_rows

            cell_lines = meta_tbl.column("cell_line").to_pylist()
            drugs = meta_tbl.column("drug").to_pylist()
            dd_concs = meta_tbl.column("drugname_drugconc").to_pylist()

            # map global row idx -> row group idx
            rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(n_row_groups)]
            rg_boundaries = []
            acc = 0
            for sz in rg_sizes:
                rg_boundaries.append((acc, acc + sz))
                acc += sz

            # find which row groups contain >=1 matching CVCL_0546 row
            rg_needed: dict[int, str] = {}
            for i, (cl, drug) in enumerate(zip(cell_lines, drugs)):
                if cl != CELL_LINE:
                    continue
                label = classify(drug)
                if label is None:
                    continue
                for rg_idx, (lo, hi) in enumerate(rg_boundaries):
                    if lo <= i < hi:
                        rg_needed[rg_idx] = True
                        break
                if label == "dmso_control":
                    seen_dmso_labels.add(drug)
                else:
                    seen_combo_drugname_drugconc[label].add(dd_concs[i])

            # targeted pass: only row groups that actually contain a match
            for rg_idx in rg_needed:
                rg_tbl = pf.read_row_group(rg_idx)
                total_row_groups_fully_read += 1
                df = rg_tbl.to_pandas()
                sub = df[df["cell_line"] == CELL_LINE]
                labels = sub["drug"].map(classify)
                for label in buckets:
                    hit = sub[labels == label]
                    if len(hit):
                        buckets[label].append(pa.Table.from_pandas(hit, preserve_index=False))

        dt = time.time() - t0
        print(
            f"shard {shard_idx:3d}/115  rows={meta_tbl.num_rows:6d}  "
            f"row_groups_needed={len(rg_needed)}/{n_row_groups}  ({dt:.1f}s)",
            flush=True,
        )

    total_time = time.time() - t_start
    print(f"\nDone scanning all {N_SHARDS} shards in {total_time:.1f}s "
          f"({total_rows_scanned} rows scanned via cheap columns, "
          f"{total_row_groups_fully_read} row groups fully read).")

    counts = {}
    for label, tables in buckets.items():
        if tables:
            merged = pa.concat_tables(tables)
        else:
            merged = None
        counts[label] = merged.num_rows if merged is not None else 0
        out_path = OUT_DIR / f"{label}.parquet"
        if merged is not None:
            pq.write_table(merged, out_path)
            print(f"wrote {out_path} : {merged.num_rows} rows")
        else:
            print(f"WARNING: no rows found for {label}, nothing written")

    print("\nDMSO 'drug' labels seen for CVCL_0546:", sorted(seen_dmso_labels))
    for combo, dd_set in seen_combo_drugname_drugconc.items():
        print(f"\n{combo} drugname_drugconc strings seen for CVCL_0546:")
        for s in sorted(dd_set):
            print(" ", s)

    print("\nCell counts:", counts)


if __name__ == "__main__":
    main()
