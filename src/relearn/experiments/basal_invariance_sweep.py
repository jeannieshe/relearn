"""
Is STATE's basal-invariance general, or specific to palbociclib/venetoclax?

real_basal_order.py found that predicting drug B from real A-treated cells gives
the same answer as predicting B from untreated controls (cos 0.967, 0.33x the
split-half noise floor) -- the model discards the basal. That rests on one pair.
This runs the reduced version of that test over N random pairs and reports the
distribution.

Per pair, per draw, only two model calls are needed:

    ctrl_to_B   = STATE(DMSO controls,      pert B)   "B alone"
    realA_to_B  = STATE(real A-treated,     pert B)   "A then B"

and the question is whether those two land on top of each other, measured against
the DMSO split-half floor (no model call -- two disjoint draws from the real
control pool). Also records ctrl->B vs real_B, so the pass-1 accuracy distribution
comes along for free.

Reading it:
  * all pairs clustered near cos ~0.97, well under 1.0x floor
        -> basal-invariance is architectural, the project-level conclusion holds
  * a spread, some pairs at 0.6-0.7
        -> the basal sometimes carries through, and *which* drugs do becomes the
           interesting question

Run with:
    python src/relearn/experiments/basal_invariance_sweep.py --n-pairs 30 --n-draws 5
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
    rel_residual,
)

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "artifacts"


def eligible_drugs(cats, counts, dose, min_cells, control_label):
    """Perturbation labels at the requested dose with enough real cells to sample."""
    out = []
    for i, c in enumerate(cats):
        if c == control_label or f", {dose}," not in c:
            continue
        if counts[i] >= min_cells:
            out.append(c)
    return out


def read_pool(dset, codes, cat_idx, max_cells, rng):
    rows = np.where(codes == cat_idx)[0]
    if len(rows) > max_cells:
        rows = rng.choice(rows, size=max_cells, replace=False)
    return dset[np.sort(rows), :].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=30)
    ap.add_argument("--n-draws", type=int, default=5)
    ap.add_argument("--dose", type=float, default=0.5)
    ap.add_argument("--sentence-len", type=int, default=256)
    ap.add_argument("--min-cells", type=int, default=600,
                    help="skip drugs with fewer real cells than this at the given dose")
    ap.add_argument("--max-cells", type=int, default=1024, help="cells loaded per drug")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = EnvConfig()
    if cfg.embed_key != "X_hvg":
        raise SystemExit(f"embed_key={cfg.embed_key!r}: this script assumes gene-space output.")
    S = args.sentence_len
    rng = np.random.default_rng(args.seed)

    h5_path = find_cell_line_file(Path(cfg.tahoe_se_dir), cfg.cell_type_accession_number)
    tm = StateTransitionModel(cfg)

    with h5py.File(h5_path, "r") as f:
        cats = [c.decode() if isinstance(c, bytes) else c
                for c in f["obs"]["drugname_drugconc"]["categories"][:]]
        codes = f["obs"]["drugname_drugconc"]["codes"][:]
        counts = np.bincount(codes, minlength=len(cats))
        dset = f["obsm"][cfg.embed_key]

        pool_names = eligible_drugs(cats, counts, args.dose, args.min_cells, cfg.dmso_control_pert)
        # 2 distinct drugs per pair, sampled without replacement, so no pool is
        # read from disk twice -- the h5ad reads dominate runtime, not the model
        need = 2 * args.n_pairs
        if len(pool_names) < need:
            raise SystemExit(f"only {len(pool_names)} eligible drugs at {args.dose}uM, need {need}")
        chosen = rng.choice(pool_names, size=need, replace=False)
        pairs = [(chosen[2 * i], chosen[2 * i + 1]) for i in range(args.n_pairs)]

        print(f"{h5_path.name}: {len(pool_names)} drugs eligible at {args.dose}uM "
              f"(>={args.min_cells} cells); sampling {args.n_pairs} pairs")
        print(f"S={S}  draws={args.n_draws}\n")

        dmso_pool = read_pool(dset, codes, cats.index(cfg.dmso_control_pert),
                              max(args.max_cells, 4 * S), rng)
        x0 = dmso_pool.mean(axis=0).astype(np.float64)
        pv_dmso = tm.pert_map[cfg.dmso_control_pert].float()

        rows = []
        for i, (name_a, name_b) in enumerate(pairs):
            pool_a = read_pool(dset, codes, cats.index(name_a), args.max_cells, rng)
            pool_b = read_pool(dset, codes, cats.index(name_b), args.max_cells, rng)
            pv_b = tm.pert_map[name_b].float()

            acc = {k: [] for k in ("cos_key", "rel_key", "e_key", "floor_rel", "floor_e",
                                   "cos_pass1", "rel_pass1", "cos_direct", "rel_direct")}
            for _ in range(args.n_draws):
                def draw(pool, n=S):
                    idx = rng.choice(len(pool), size=n, replace=len(pool) < n)
                    return pool[idx].astype(np.float64)

                basal_ctrl, basal_a = draw(dmso_pool), draw(pool_a)
                real_a, real_b = draw(pool_a), draw(pool_b)
                dmso_2S = draw(dmso_pool, 2 * S)

                P = {
                    "ctrl_to_B": predict(tm, basal_ctrl, pv_b),
                    "realA_to_B": predict(tm, basal_a, pv_b),
                    "real_A": real_a, "real_B": real_b,
                    "h1": dmso_2S[:S], "h2": dmso_2S[S:],
                }
                V = {k: p.mean(axis=0) - x0 for k, p in P.items()}
                ref = 0.5 * (np.linalg.norm(V["real_A"]) + np.linalg.norm(V["real_B"]))

                acc["cos_key"].append(cos(V["realA_to_B"], V["ctrl_to_B"]))
                acc["rel_key"].append(rel_residual(V["realA_to_B"], V["ctrl_to_B"], ref))
                acc["e_key"].append(energy_distance(P["realA_to_B"], P["ctrl_to_B"]))
                acc["floor_rel"].append(rel_residual(V["h1"], V["h2"], ref))
                acc["floor_e"].append(energy_distance(P["h1"], P["h2"]))
                acc["cos_pass1"].append(cos(V["ctrl_to_B"], V["real_B"]))
                acc["rel_pass1"].append(rel_residual(V["ctrl_to_B"], V["real_B"], ref))
                acc["cos_direct"].append(cos(V["realA_to_B"], V["real_B"]))
                acc["rel_direct"].append(rel_residual(V["realA_to_B"], V["real_B"], ref))

            m = {k: float(np.mean(v)) for k, v in acc.items()}
            rows.append({
                "pair_index": i,
                "drug_a": name_a,
                "drug_b": name_b,
                "cos_realAtoB_vs_ctrlB": m["cos_key"],
                "rel_realAtoB_vs_ctrlB": m["rel_key"],
                "x_floor_rel": m["rel_key"] / m["floor_rel"] if m["floor_rel"] else float("nan"),
                "x_floor_energy": m["e_key"] / m["floor_e"] if m["floor_e"] else float("nan"),
                "cos_ctrlB_vs_realB": m["cos_pass1"],
                "x_floor_pass1": m["rel_pass1"] / m["floor_rel"] if m["floor_rel"] else float("nan"),
                "cos_realAtoB_vs_realB": m["cos_direct"],
                "x_floor_direct": m["rel_direct"] / m["floor_rel"] if m["floor_rel"] else float("nan"),
                "n_cells_a": int(counts[cats.index(name_a)]),
                "n_cells_b": int(counts[cats.index(name_b)]),
            })
            print(f"  [{i+1:>2}/{args.n_pairs}] cos={m['cos_key']:.4f} "
                  f"({rows[-1]['x_floor_rel']:.2f}x floor)  "
                  f"{str(name_a)[2:26]:<26} -> {str(name_b)[2:26]}")

    tag = args.tag or f"{args.n_pairs}pairs_{args.dose}uM_S{S}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"basal_invariance_sweep_{tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def dist(field):
        v = np.array([r[field] for r in rows], dtype=float)
        return (f"median={np.median(v):.4f}  IQR=[{np.quantile(v,.25):.4f}, {np.quantile(v,.75):.4f}]  "
                f"min={v.min():.4f}  max={v.max():.4f}")

    print(f"\n--- distribution over {len(rows)} pairs ---")
    print(f"cos(realA->B, ctrl->B)      {dist('cos_realAtoB_vs_ctrlB')}")
    print(f"  as x noise floor (resid)  {dist('x_floor_rel')}")
    print(f"  as x noise floor (energy) {dist('x_floor_energy')}")
    print(f"cos(ctrl->B,   real_B)      {dist('cos_ctrlB_vs_realB')}   [pass-1 accuracy]")
    print(f"cos(realA->B,  real_B)      {dist('cos_realAtoB_vs_realB')}   [direct seq-vs-single]")

    xf = np.array([r["x_floor_rel"] for r in rows], dtype=float)
    n_inv = int((xf < 1.0).sum())
    print(f"\npairs where A->B is indistinguishable from B alone (< 1.0x floor): "
          f"{n_inv}/{len(rows)}")
    print("verdict: " + (
        "BASAL-INVARIANCE IS GENERAL -- not specific to palbociclib/venetoclax"
        if n_inv == len(rows) else
        f"MIXED -- {len(rows)-n_inv} pair(s) clear the floor; inspect those rows in {out_csv.name}"))
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
