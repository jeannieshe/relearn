"""
Dose-matching table between EmeraldBay's measured doses (for the 14 single
drugs + 2 combos that have a full name match in STATE's Tahoe-100M panel --
see artifacts/emeraldbay_single_drug_matches.csv / emeraldbay_combo_matches.csv)
and whatever (name, concentration, units) entries actually exist in
STATE's pert_onehot_map.pt.

A drug name matching Tahoe-100M is necessary but not sufficient: STATE's
action space is per EXACT dose, not per drug, so env.step()/multi_hot() can
only apply whatever concentration Tahoe-100M happened to screen that drug at
-- which will essentially never be the exact concentration EmeraldBay used.
This script picks the nearest available Tahoe-100M dose (same units) for
each EmeraldBay condition and records how far off it is, so the eventual
validation experiment uses a documented, reproducible dose choice instead of
an implicit/arbitrary one.

Run with:
    /home/jeannie/miniconda/envs/pytorch-pip/bin/python \
        src/relearn/experiments/emeraldbay_dose_matching.py
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent.parent.parent.parent
PERT_MAP_PATH = REPO_ROOT / "data/models/ST-HVG-Tahoe/fewshot/state_generalization_X_hvg/pert_onehot_map.pt"
OUT_PATH = REPO_ROOT / "artifacts/emeraldbay_dose_matching.csv"
EMERALDBAY_REPO = "tahoebio/EmeraldBay"
CELL_LINE = "CVCL_0546"

# EmeraldBay name -> Tahoe-100M pert_map name, from artifacts/emeraldbay_single_drug_matches.csv
EB_TO_TAHOE_NAME = {
    "Adagrasib": "Adagrasib",
    "Dabrafenib": "Dabrafenib",
    "Encorafenib": "Encorafenib",
    "Fluorouracil": "5-Fluorouracil",
    "Gemcitabine": "Gemcitabine",
    "Irinotecan": "Irinotecan",
    "Lapatinib ditosylate": "Lapatinib ditosylate",
    "Oxaliplatin": "Oxaliplatin",
    "Paclitaxel": "Paclitaxel",
    "RMC-6236": "RMC-6236",
    "Regorafenib": "Regorafenib",
    "Trametinib": "Trametinib",
    "Trifluridine": "Trifluridine",
    "Tucatinib": "Tucatinib",
}
COMBO_CONDITIONS = ["Gemcitabine+Paclitaxel", "Dabrafenib+Trametinib"]


def load_tahoe_doses() -> dict[str, list[tuple[float, str]]]:
    """Tahoe-100M name -> list of (concentration, unit) pairs available in pert_onehot_map.pt."""
    pert_map = torch.load(PERT_MAP_PATH, weights_only=False)
    doses: dict[str, list[tuple[float, str]]] = {}
    for key in pert_map.keys():
        entries = ast.literal_eval(key) if isinstance(key, str) else key
        for name, conc, unit in entries:
            doses.setdefault(name, []).append((float(conc), unit))
    return doses


def load_emeraldbay_sw480_conditions() -> list[list[tuple[str, float, str]]]:
    """Every distinct `condition` (list of (drug, conc, unit) tuples) tested on
    SW480, from metadata/summary_statistics.parquet -- the same exhaustive
    index emeraldbay_overlap_check.py used for the combo list."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(repo_id=EMERALDBAY_REPO, repo_type="dataset", filename="metadata/summary_statistics.parquet")
    df = pq.read_table(path).to_pandas()
    sw480 = df[df["cell_line"] == CELL_LINE]
    conditions = []
    for cond in sw480["condition"]:
        parsed = ast.literal_eval(cond) if isinstance(cond, str) else cond
        conditions.append(parsed)
    return conditions


def nearest_dose(target_conc: float, target_unit: str, available: list[tuple[float, str]]):
    same_unit = [(c, u) for c, u in available if u == target_unit]
    if not same_unit:
        return None
    conc, unit = min(same_unit, key=lambda cu: abs(cu[0] - target_conc))
    return conc, unit


def main():
    tahoe_doses = load_tahoe_doses()
    conditions = load_emeraldbay_sw480_conditions()

    rows = []

    # single drugs: pull every (drug, conc, unit) triple for our 14 targets
    # out of every SW480 condition list (single-drug conditions are length-1
    # lists; the same drug can also appear as a component inside a combo
    # condition, so filter to length-1 to keep this table single-drug-only)
    single_doses_seen: dict[str, set[tuple[float, str]]] = {d: set() for d in EB_TO_TAHOE_NAME}
    for cond in conditions:
        if len(cond) == 1:
            name, conc, unit = cond[0]
            if name in single_doses_seen:
                single_doses_seen[name].add((float(conc), unit))

    for eb_name, tahoe_name in EB_TO_TAHOE_NAME.items():
        available = tahoe_doses.get(tahoe_name, [])
        eb_doses = sorted(single_doses_seen[eb_name])
        if not eb_doses:
            rows.append({
                "drug": eb_name, "emeraldbay_dose": "", "emeraldbay_unit": "",
                "tahoe100m_matched_conc": "", "tahoe100m_matched_unit": "",
                "exact_match": "", "pert_map_key": "",
                "note": "NO SW480 single-drug condition found in summary_statistics.parquet",
            })
            continue
        for conc, unit in eb_doses:
            if not available:
                rows.append({
                    "drug": eb_name, "emeraldbay_dose": conc, "emeraldbay_unit": unit,
                    "tahoe100m_matched_conc": "", "tahoe100m_matched_unit": "",
                    "exact_match": False, "pert_map_key": "",
                    "note": f"NAME MATCHES ({tahoe_name!r}) BUT ZERO DOSES IN pert_onehot_map.pt -- unusable",
                })
                continue
            match = nearest_dose(conc, unit, available)
            if match is None:
                rows.append({
                    "drug": eb_name, "emeraldbay_dose": conc, "emeraldbay_unit": unit,
                    "tahoe100m_matched_conc": "", "tahoe100m_matched_unit": "",
                    "exact_match": False, "pert_map_key": "",
                    "note": f"no Tahoe-100M dose in matching units (available units: {sorted(set(u for _, u in available))})",
                })
                continue
            m_conc, m_unit = match
            rows.append({
                "drug": eb_name, "emeraldbay_dose": conc, "emeraldbay_unit": unit,
                "tahoe100m_matched_conc": m_conc, "tahoe100m_matched_unit": m_unit,
                "exact_match": abs(m_conc - conc) < 1e-9,
                "pert_map_key": str([(tahoe_name, m_conc, m_unit)]),
                "note": "",
            })

    # combos: reuse the pipe-delimited component doses already known from the
    # emeraldbay_sw480_pull.py report, cross-checked against summary_statistics
    combo_doses_seen: dict[str, set[tuple]] = {c: set() for c in COMBO_CONDITIONS}
    for cond in conditions:
        if len(cond) < 2:
            continue
        names = {n for n, _, _ in cond}
        for combo in COMBO_CONDITIONS:
            combo_names = set(combo.split("+"))
            if names == combo_names:
                combo_doses_seen[combo].add(tuple(sorted(cond)))

    for combo, dose_sets in combo_doses_seen.items():
        for dose_set in sorted(dose_sets):
            per_component_matches = []
            for name, conc, unit in dose_set:
                tahoe_name = EB_TO_TAHOE_NAME.get(name, name)
                available = tahoe_doses.get(tahoe_name, [])
                match = nearest_dose(float(conc), unit, available)
                per_component_matches.append((name, tahoe_name, conc, unit, match))
            note_parts = []
            key_parts = []
            all_matched = True
            for name, tahoe_name, conc, unit, match in per_component_matches:
                if match is None:
                    all_matched = False
                    note_parts.append(f"{name}: NO MATCH")
                else:
                    m_conc, m_unit = match
                    exact = abs(m_conc - float(conc)) < 1e-9
                    note_parts.append(f"{name}: eb={conc}{unit} -> tahoe={m_conc}{m_unit} ({'exact' if exact else 'nearest'})")
                    key_parts.append((tahoe_name, m_conc, m_unit))
            rows.append({
                "drug": combo, "emeraldbay_dose": "; ".join(f"{n}={c}{u}" for n, c, u in dose_set),
                "emeraldbay_unit": "", "tahoe100m_matched_conc": "", "tahoe100m_matched_unit": "",
                "exact_match": all(abs(m[4][0] - float(m[2])) < 1e-9 for m in per_component_matches if m[4]),
                "pert_map_key": str(key_parts) if all_matched else "",
                "note": "; ".join(note_parts),
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "drug", "emeraldbay_dose", "emeraldbay_unit", "tahoe100m_matched_conc",
            "tahoe100m_matched_unit", "exact_match", "pert_map_key", "note",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_PATH}\n")
    n_exact = sum(1 for r in rows if r["exact_match"] is True)
    n_nearest = sum(1 for r in rows if r["exact_match"] is False and r["pert_map_key"])
    n_unusable = sum(1 for r in rows if not r["pert_map_key"])
    print(f"exact matches: {n_exact}, nearest-dose matches: {n_nearest}, unusable (no dose available): {n_unusable}")
    print("\nUnusable rows (name matches Tahoe-100M but no usable dose found):")
    for r in rows:
        if not r["pert_map_key"]:
            print(f"  {r['drug']}: {r['note']}")


if __name__ == "__main__":
    main()
