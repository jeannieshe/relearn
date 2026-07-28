"""
Targeted pull of SW480 (CVCL_0546) rows from tahoebio/EmeraldBay's
`expression_data` (116 parquet shards, ~57.7 GB total) for the 14 SINGLE-drug
treatments that have a name match in STATE's Tahoe-100M panel (see
artifacts/emeraldbay_single_drug_matches.csv / emeraldbay_overlap_summary.json).

This is a sibling of emeraldbay_sw480_pull.py (which pulled DMSO controls +
2 combo conditions) -- same row-group-level parquet filtering approach
(cheap columns first to find which row groups contain a match, then only
`read_row_group` those), extended to classify against 14 target single-drug
labels in ONE pass over all 116 shards instead of one scan per drug.

Target drugs (EmeraldBay's own `drug` column spelling, from
artifacts/emeraldbay_single_drug_matches.csv's `emeraldbay_drug` column --
note this is EmeraldBay's own name, e.g. "Fluorouracil", not Tahoe-100M's
matched name "5-Fluorouracil"):
    Adagrasib, Dabrafenib, Encorafenib, Fluorouracil, Gemcitabine,
    Irinotecan, "Lapatinib ditosylate", Oxaliplatin, Paclitaxel, RMC-6236,
    Regorafenib, Trametinib, Trifluridine, Tucatinib

A "single-drug" row is matched by exact equality of the `drug` column against
one of these names (single-drug rows never contain "+"; combo rows are
"DrugA+DrugB[+DrugC...]"). All dose variants (distinct `drugname_drugconc`
strings) found per drug are kept -- no pre-filtering to one dose.

Output: one parquet per drug under data/datasets/EmeraldBay/SW480/singles/,
same raw EmeraldBay row schema as the existing dmso_control/combo pulls
(genes, expressions, drug, drugname_drugconc, cell_line, sample,
BARCODE_SUB_LIB_ID) -- no gene-panel mapping or normalization here.

Run with:
    /home/jeannie/miniconda/envs/pytorch-pip/bin/python \
        src/relearn/experiments/emeraldbay_sw480_singles_pull.py
"""

import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "data/datasets/EmeraldBay/SW480/singles"
EMERALDBAY_REPO = "tahoebio/EmeraldBay"
N_SHARDS = 116
CELL_LINE = "CVCL_0546"

# EmeraldBay's own `drug` column spelling (NOT Tahoe-100M's matched name --
# e.g. "Fluorouracil" here, "5-Fluorouracil" is Tahoe-100M's name for the
# same compound). Order matches artifacts/emeraldbay_single_drug_matches.csv.
TARGET_DRUGS = [
    "Adagrasib",
    "Dabrafenib",
    "Encorafenib",
    "Fluorouracil",
    "Gemcitabine",
    "Irinotecan",
    "Lapatinib ditosylate",
    "Oxaliplatin",
    "Paclitaxel",
    "RMC-6236",
    "Regorafenib",
    "Trametinib",
    "Trifluridine",
    "Tucatinib",
]


def sanitize(name: str) -> str:
    out = []
    for ch in name.strip().lower():
        out.append(ch if ch.isalnum() else "_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


DRUG_TO_FILE = {d: sanitize(d) for d in TARGET_DRUGS}
TARGET_SET = set(TARGET_DRUGS)


def classify(drug_str: str) -> str | None:
    d = drug_str.strip()
    return d if d in TARGET_SET else None


def shard_path(idx: int) -> str:
    return (
        f"datasets/{EMERALDBAY_REPO}/expression_data/"
        f"train-{idx:05d}-of-{N_SHARDS:05d}.parquet"
    )


def main():
    fs = HfFileSystem()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write incrementally (one ParquetWriter per drug, flushed after every
    # shard) instead of buffering all 115 shards' matches in memory and
    # writing only at the end -- two prior long-running EmeraldBay pulls in
    # this environment died mid-scan (once from a harness restart, once for
    # an unconfirmed reason with no OOM/kill evidence in dmesg despite ample
    # free memory). Buffer-to-the-end meant both losses were total; this way
    # a crash at shard N still leaves shards 0..N-1's matches usable on disk,
    # and resuming just means re-scanning (cheap: ~7-10s/shard for the cheap
    # columns) rather than redoing everything.
    writers: dict[str, pq.ParquetWriter | None] = {d: None for d in TARGET_DRUGS}
    seen_drugname_drugconc: dict[str, set[str]] = {d: set() for d in TARGET_DRUGS}
    unexpected_labels_seen: set[str] = set()
    counts: dict[str, int] = {d: 0 for d in TARGET_DRUGS}

    t_start = time.time()
    total_rows_scanned = 0
    total_row_groups_fully_read = 0

    last_shard_scanned = -1
    try:
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

                # find which row groups contain >=1 matching CVCL_0546 row,
                # checking membership against ALL 14 target drugs simultaneously
                # (single pass over cheap columns, not 14 separate scans)
                rg_needed: dict[int, bool] = {}
                for i, (cl, drug) in enumerate(zip(cell_lines, drugs)):
                    if cl != CELL_LINE:
                        continue
                    label = classify(drug)
                    if label is None:
                        # note anything that looks SW480 + a near-miss spelling
                        # of a target drug (case/whitespace variant) for the report
                        stripped = drug.strip()
                        for td in TARGET_DRUGS:
                            if stripped.lower() == td.lower() and stripped != td:
                                unexpected_labels_seen.add(drug)
                        continue
                    for rg_idx, (lo, hi) in enumerate(rg_boundaries):
                        if lo <= i < hi:
                            rg_needed[rg_idx] = True
                            break
                    seen_drugname_drugconc[label].add(dd_concs[i])

                # targeted pass: only row groups that actually contain a match.
                # Write each match straight to its drug's ParquetWriter (opened
                # lazily on first match) instead of buffering in memory -- see
                # the note above main() for why.
                for rg_idx in rg_needed:
                    rg_tbl = pf.read_row_group(rg_idx)
                    total_row_groups_fully_read += 1
                    df = rg_tbl.to_pandas()
                    sub = df[df["cell_line"] == CELL_LINE]
                    labels = sub["drug"].map(classify)
                    for label in TARGET_DRUGS:
                        hit = sub[labels == label]
                        if len(hit):
                            hit_tbl = pa.Table.from_pandas(hit, preserve_index=False)
                            if writers[label] is None:
                                out_path = OUT_DIR / f"{DRUG_TO_FILE[label]}.parquet"
                                writers[label] = pq.ParquetWriter(out_path, hit_tbl.schema)
                            writers[label].write_table(hit_tbl)
                            counts[label] += hit_tbl.num_rows

            last_shard_scanned = shard_idx
            dt = time.time() - t0
            print(
                f"shard {shard_idx:3d}/{N_SHARDS - 1}  rows={meta_tbl.num_rows:6d}  "
                f"row_groups_needed={len(rg_needed)}/{n_row_groups}  ({dt:.1f}s)",
                flush=True,
            )
    finally:
        # close whatever writers were opened so every parquet written so far
        # is valid and readable, even if the loop above raised or was killed
        for label, w in writers.items():
            if w is not None:
                w.close()

    total_time = time.time() - t_start
    completed = last_shard_scanned == N_SHARDS - 1
    print(f"\n{'Done scanning' if completed else 'STOPPED (did not finish) after'} "
          f"{last_shard_scanned + 1}/{N_SHARDS} shards in {total_time:.1f}s "
          f"({total_rows_scanned} rows scanned via cheap columns, "
          f"{total_row_groups_fully_read} row groups fully read).")
    if not completed:
        print(f"WARNING: incomplete scan -- counts below only reflect shards "
              f"0..{last_shard_scanned}, not the full dataset. Re-run to continue "
              f"coverage (already-written rows are valid, just partial).")

    for label in TARGET_DRUGS:
        if counts[label] == 0:
            out_path = OUT_DIR / f"{DRUG_TO_FILE[label]}.parquet"
            print(f"WARNING: no rows found for {label!r} (in shards scanned so far), "
                  f"nothing written (out_path would have been {out_path})")

    print("\nDistinct drugname_drugconc (dose) strings seen per drug, for CVCL_0546:")
    for d in TARGET_DRUGS:
        print(f"\n{d} ({counts[d]} rows):")
        for s in sorted(seen_drugname_drugconc[d]):
            print(" ", s)

    if unexpected_labels_seen:
        print("\nWARNING: near-miss `drug` label spellings seen (case/whitespace "
              "variants of a target name, not counted above):", sorted(unexpected_labels_seen))
    else:
        print("\nNo near-miss/unexpected label spellings seen -- all matches were exact.")

    print("\nCell counts:", counts)


if __name__ == "__main__":
    main()
