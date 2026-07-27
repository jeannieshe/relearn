"""
Does the basal tensor actually reach STATE's forward pass, and how much can it
move the output?

basal_invariance_sweep.py showed that swapping the basal from DMSO controls to
real drug-treated cells barely changes the prediction (cos 0.971, 0.31x the noise
floor, 30/30 pairs). Two readings of that:

  (a) learned invariance -- the basal is read but weighted weakly
  (b) plumbing bug       -- the basal never reaches forward at all, and the model
                            is a lookup table from perturbation one-hot to a
                            memorized average response

This distinguishes them. The perturbation is held FIXED and only the basal slot
varies, so any change in output can only have come from the basal. All conditions
stay within SW480, and run from "real, biologically graded" to outright garbage:

    dmso           real DMSO controls                       <- reference
    dmso_b         independent second draw of the same      <- NOISE FLOOR
    palbo_lo       real palbociclib-treated cells, 0.5 uM
    palbo_hi       real palbociclib-treated cells, 5.0 uM   (bigger real change)
    veneto         real venetoclax-treated cells, 0.5 uM
    shuffled       per-cell permutation of gene values -- keeps each cell's exact
                   values, sparsity and library size, destroys gene identity
    gaussian       per-gene mean/std of the real DMSO pool, sampled independently
                   -- keeps marginals, destroys gene-gene correlation
    zeros          basal absent

Reported as GAIN: how far the output moved per unit of how far the input moved,
both in units of the split-half floor. That is the number the two readings differ
on. If garbage input moves the output no further than real cells do, the basal's
whole influence budget sits below measurement noise (reading a, strong form). If
garbage moves it well past the floor, the model does read the basal and simply
learned that realistic variation does not matter -- reading (b) is dead, but so is
the strong form of the invariance claim.

Run with:
    python src/relearn/experiments/basal_control_sweep.py --n-drugs 6 --n-draws 5
"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from relearn.config import EnvConfig
from relearn.transitions.state_model import StateTransitionModel
from relearn.experiments.real_basal_order import (
    cos,
    energy_distance,
    find_cell_line_file,
    predict,
    resolve_label,
)
from relearn.experiments.basal_invariance_sweep import eligible_drugs, read_pool

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "artifacts"

# real drug-treated basals, all SW480. dose-graded so there is a range of genuine
# biological change to compare the garbage conditions against.
REAL_BASALS = [("palbo_lo", "palbociclib", 0.5),
               ("palbo_hi", "palbociclib", 5.0),
               ("veneto", "venetoclax", 0.5)]

CONDITIONS = ["dmso_b", "palbo_lo", "palbo_hi", "veneto", "shuffled", "gaussian", "zeros"]


def make_basals(pools, S, rng):
    """One draw of every basal condition. Same shape, wildly different content."""
    def draw(pool, n=S):
        idx = rng.choice(len(pool), size=n, replace=len(pool) < n)
        return pool[idx].astype(np.float64)

    dmso = pools["dmso"]
    # marginal-matched noise: right per-gene mean/std, no gene-gene structure
    mu, sd = dmso.mean(0).astype(np.float64), dmso.std(0).astype(np.float64)
    gauss = np.clip(rng.normal(mu, sd, size=(S, dmso.shape[1])), 0, None)
    # per-cell permutation: identical values, wrong genes
    shuf = np.stack([row[rng.permutation(row.shape[0])] for row in draw(dmso)])

    basals = {
        "dmso": draw(dmso),
        "dmso_b": draw(dmso),
        "shuffled": shuf,
        "gaussian": gauss,
        "zeros": np.zeros((S, dmso.shape[1]), dtype=np.float64),
    }
    for key, _name, _dose in REAL_BASALS:
        basals[key] = draw(pools[key])
    return basals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-drugs", type=int, default=6,
                    help="random perturbations, on top of palbociclib and venetoclax")
    ap.add_argument("--n-draws", type=int, default=5)
    ap.add_argument("--dose", type=float, default=0.5, help="dose for the perturbation arm")
    ap.add_argument("--sentence-len", type=int, default=256)
    ap.add_argument("--max-cells", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = EnvConfig()
    if cfg.embed_key != "X_hvg":
        raise SystemExit(f"embed_key={cfg.embed_key!r}: this script assumes gene-space output.")
    S = args.sentence_len
    rng = np.random.default_rng(args.seed)
    tm = StateTransitionModel(cfg)

    h5_path = find_cell_line_file(Path(cfg.tahoe_se_dir), cfg.cell_type_accession_number)
    pools = {}
    with h5py.File(h5_path, "r") as f:
        cats = [c.decode() if isinstance(c, bytes) else c
                for c in f["obs"]["drugname_drugconc"]["categories"][:]]
        codes = f["obs"]["drugname_drugconc"]["codes"][:]
        counts = np.bincount(codes, minlength=len(cats))
        dset = f["obsm"][cfg.embed_key]

        pools["dmso"] = read_pool(dset, codes, cats.index(cfg.dmso_control_pert),
                                  max(args.max_cells, 4 * S), rng)
        for key, name, dose in REAL_BASALS:
            label = resolve_label(cats, name, dose)
            pools[key] = read_pool(dset, codes, cats.index(label), args.max_cells, rng)
            print(f"  basal {key:<10} {label:<40} n={counts[cats.index(label)]}")

        elig = eligible_drugs(cats, counts, args.dose, 1, cfg.dmso_control_pert)
        fixed = [resolve_label(cats, n, args.dose) for n in ("palbociclib", "venetoclax")]
        perts = fixed + list(rng.choice([e for e in elig if e not in fixed],
                                        size=args.n_drugs, replace=False))

    x0 = pools["dmso"].mean(axis=0).astype(np.float64)
    print(f"\ncell line: {cfg.cell_type_name} ({h5_path.name})   "
          f"perturbations: {len(perts)}   S={S}   draws={args.n_draws}\n")

    rows, zero_maxdiff = [], []
    for di, pert_label in enumerate(perts):
        pv = tm.pert_map[pert_label].float()
        acc = {c: {"cos": [], "out_e": [], "in_e": []} for c in CONDITIONS}
        for _ in range(args.n_draws):
            basals = make_basals(pools, S, rng)
            ref_in = basals["dmso"]
            ref_out = predict(tm, ref_in, pv)
            v_ref = ref_out.mean(0) - x0
            for c in CONDITIONS:
                out = predict(tm, basals[c], pv)
                acc[c]["cos"].append(cos(out.mean(0) - x0, v_ref))
                acc[c]["out_e"].append(energy_distance(out, ref_out))
                acc[c]["in_e"].append(energy_distance(basals[c], ref_in))
                if c == "zeros":
                    zero_maxdiff.append(float(np.abs(out - ref_out).max()))

        floor_in = float(np.mean(acc["dmso_b"]["in_e"]))
        floor_out = float(np.mean(acc["dmso_b"]["out_e"]))
        for c in CONDITIONS:
            in_x = float(np.mean(acc[c]["in_e"])) / floor_in
            out_x = float(np.mean(acc[c]["out_e"])) / floor_out
            rows.append({
                "pert": pert_label, "basal_condition": c,
                "cos_vs_reference_output": float(np.mean(acc[c]["cos"])),
                "input_change_x_floor": in_x,
                "output_change_x_floor": out_x,
                "gain_out_per_in": out_x / in_x if in_x > 0 else float("nan"),
            })
        print(f"  [{di+1}/{len(perts)}] {str(pert_label)[2:32]:<32} "
              + " ".join(f"{c}:{np.mean(acc[c]['out_e'])/floor_out:.2f}x" for c in CONDITIONS[1:]))

    tag = args.tag or f"{len(perts)}perts_{args.dose}uM_S{S}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"basal_control_sweep_{tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n--- averaged over {len(perts)} perturbations (units: x split-half floor) ---")
    print(f"{'basal condition':<16}{'input moved':>13}{'output moved':>14}{'gain':>8}{'cos to ref':>12}")
    for c in CONDITIONS:
        sub = [r for r in rows if r["basal_condition"] == c]
        print(f"{c:<16}{np.mean([r['input_change_x_floor'] for r in sub]):>13.2f}"
              f"{np.mean([r['output_change_x_floor'] for r in sub]):>14.2f}"
              f"{np.mean([r['gain_out_per_in'] for r in sub]):>8.3f}"
              f"{np.mean([r['cos_vs_reference_output'] for r in sub]):>12.4f}")

    print(f"\nzeros-vs-reference max |difference| per gene: {max(zero_maxdiff):.6f}")
    print("  (exactly 0.0 would mean the basal tensor never reaches forward)")

    garbage = np.mean([r["output_change_x_floor"] for r in rows
                       if r["basal_condition"] in ("shuffled", "gaussian", "zeros")])
    real_bio = np.mean([r["output_change_x_floor"] for r in rows
                        if r["basal_condition"] in ("palbo_lo", "palbo_hi", "veneto")])
    print(f"\nreal drug-treated basal moves output: {real_bio:.2f}x floor")
    print(f"outright garbage moves output:        {garbage:.2f}x floor")
    print("verdict: " + (
        "BASAL BARELY REACHES OUTPUT -- even garbage stays near the noise floor, so the "
        "influence budget is too small for real biology to ever register"
        if garbage < 2.0 else
        "BASAL IS READ AND HAS RANGE -- garbage moves the output well past the floor, so the "
        "model actively learned that realistic basal variation does not matter"))
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
