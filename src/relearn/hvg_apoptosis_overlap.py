"""
How much of the HALLMARK_APOPTOSIS signature is actually representable in
STATE's 2000-HVG panel? ucell_score can only see signature genes that appear
as columns in X_hvg -- this quantifies how much of the reward signal's
biological basis survives that projection.
"""

import csv
from pathlib import Path

import numpy as np

from relearn.utils import _load_gmt_signature

HVG_GENE_NAMES_PATH = Path("/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy")
GMT_PATH = Path("/home/jeannie/relearn/data/HALLMARK_APOPTOSIS.v2026.1.Hs.gmt")
SIGNATURE_NAME = "HALLMARK_APOPTOSIS"


def compute_overlap():
    hvg_genes = np.load(HVG_GENE_NAMES_PATH, allow_pickle=True).astype(str)
    sig_genes = _load_gmt_signature(str(GMT_PATH), SIGNATURE_NAME)

    hvg_set = set(hvg_genes)
    sig_set = set(sig_genes)
    overlap = sig_set & hvg_set
    missing = sig_set - hvg_set

    return {
        "hvg_panel_size": len(hvg_set),
        "signature_size": len(sig_set),
        "overlap_count": len(overlap),
        "overlap_pct": len(overlap) / len(sig_set) * 100,
        "overlap_genes": sorted(overlap),
        "missing_genes": sorted(missing),
    }


if __name__ == "__main__":
    result = compute_overlap()

    out_path = Path(__file__).parent.parent.parent / "data" / "hvg_apoptosis_overlap.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gene", "in_hvg_panel"])
        for gene in result["overlap_genes"]:
            writer.writerow([gene, True])
        for gene in result["missing_genes"]:
            writer.writerow([gene, False])

    print(f"HVG panel size: {result['hvg_panel_size']}")
    print(f"HALLMARK_APOPTOSIS signature size: {result['signature_size']}")
    print(f"Overlap: {result['overlap_count']} genes ({result['overlap_pct']:.1f}%)")
    print(f"Wrote per-gene table to {out_path}")
