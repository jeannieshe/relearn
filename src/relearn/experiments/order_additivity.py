"""
Three-arm comparison for a drug pair (A, B), in gene-expression space rather
than through the scalar reward:

  forward      x0 -> A -> B
  reverse      x0 -> B -> A
  co-perturb   x0 -> (A and B simultaneously, one step, combined pert_emb)

against the additive null  v_A + v_B  built from the two single-drug arms.

Two questions, one sweep:
  1. ORDER      is A->B actually different from B->A?
  2. ADDITIVITY is either sequential arm different from v_A + v_B, and is
                simultaneous co-dosing different again?

Everything is measured on *displacement* vectors v = expr(after) - expr(x0),
never on raw post-perturbation states. Raw states are dominated by the shared
baseline profile, so cosine between any two of them sits at ~0.999 no matter
what the drugs did -- subtracting the common baseline is what makes the angles
mean anything. (This is why the cosine_sim column in dmso_second_sweep.py is
uninformative: it compares raw states.)

THE NULL MATTERS. A->B costs two forward passes; the co-perturbation arm costs
one. Model error accumulates per pass, so some A->B vs B->A difference is pure
integration drift, not biology. Before believing any order effect, we measure
the drift floor with wash-out arms that *should* be no-ops:

    A->DMSO vs A,  B->DMSO vs B,  DMSO->A vs A,  DMSO->B vs B

An order effect only counts if it is larger than that floor. This is the
self-consistency gate from dmso_first_sweep.py / dmso_second_sweep.py, stated
as a normalized residual on displacements and used as a reference scale.

Run with:
    python src/relearn/experiments/order_additivity.py \
        --drug-a palbociclib --drug-b venetoclax --dose 0.5
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from relearn.config import EnvConfig
from relearn.envs.small_molecules import RelearnChemicalEnv

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "artifacts"


def resolve_drug(env: RelearnChemicalEnv, name: str, dose: float):
    """Find the action key for a drug name at a given concentration."""
    matches = [
        d for d in env.drug_list
        if name.lower() in str(d).lower() and f", {dose}," in str(d)
    ]
    if not matches:
        near = [str(d) for d in env.drug_list if name.lower() in str(d).lower()]
        raise KeyError(
            f"no action for {name!r} at {dose} uM. "
            + (f"available doses: {near}" if near else f"{name!r} is not in the action space")
        )
    if len(matches) > 1:
        raise KeyError(f"{name!r} at {dose} uM is ambiguous: {[str(m) for m in matches]}")
    return matches[0]


def cos(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / denom) if denom > 0 else float("nan")


def rel_residual(u: np.ndarray, v: np.ndarray) -> float:
    """||u - v|| normalized by the average magnitude, so it reads as a fraction
    of the effect size rather than in arbitrary expression units."""
    scale = 0.5 * (np.linalg.norm(u) + np.linalg.norm(v))
    return float(np.linalg.norm(u - v) / scale) if scale > 0 else float("nan")


def run_arms(env: RelearnChemicalEnv, key_a, key_b, dmso_key):
    """
    Every arm the comparison needs. Each is an independent rollout from the same
    fixed baseline, so nothing leaks between arms.
    """
    tm = env._transition_model
    if not hasattr(tm, "step_with_pert_vector"):
        raise TypeError(
            f"transition_model={env.cfg.transition_model!r} has no step_with_pert_vector, "
            "so the co-perturbation arm can't run. Use transition_model=state."
        )

    idx_a = env.drug_list.index(key_a)
    idx_b = env.drug_list.index(key_b)
    idx_dmso = env.drug_list.index(dmso_key)

    x0_state = env.initial_cell_state
    x0 = np.asarray(env._to_gene_expression(x0_state), dtype=np.float64)

    def seq(*action_idxs):
        """Apply actions in order from the fixed baseline; return final gene-space expr."""
        s = x0_state
        for a in action_idxs:
            s = tm.step(s, a)
        return np.asarray(env._to_gene_expression(s), dtype=np.float64)

    def combo(weight: float):
        """One step with a combined perturbation vector: weight*(onehot_A + onehot_B)."""
        pv = weight * (tm.pert_map[key_a].float() + tm.pert_map[key_b].float())
        s = tm.step_with_pert_vector(x0_state, pv)
        return np.asarray(env._to_gene_expression(s), dtype=np.float64)

    arms = {
        # singles -- the building blocks of the additive null
        "A": seq(idx_a),
        "B": seq(idx_b),
        # the three arms under test
        "A_then_B": seq(idx_a, idx_b),
        "B_then_A": seq(idx_b, idx_a),
        # co-dose. mean keeps the pert_emb at single-drug magnitude (in
        # distribution); sum is the literal two-hot but doubles the input norm.
        "co_mean": combo(0.5),
        "co_sum": combo(1.0),
        # wash-out nulls -- these establish the per-pass drift floor
        "A_then_DMSO": seq(idx_a, idx_dmso),
        "B_then_DMSO": seq(idx_b, idx_dmso),
        "DMSO_then_A": seq(idx_dmso, idx_a),
        "DMSO_then_B": seq(idx_dmso, idx_b),
        "DMSO": seq(idx_dmso),
        "DMSO_then_DMSO": seq(idx_dmso, idx_dmso),
    }

    # displacements from the shared baseline
    V = {k: expr - x0 for k, expr in arms.items()}
    V["additive"] = V["A"] + V["B"]

    scores = {
        k: float(env.apoptosis_predictor(expr, gene_names=env.hvg_gene_names, signature_genes=env.sig_genes))
        for k, expr in arms.items()
    }
    scores["baseline"] = float(
        env.apoptosis_predictor(x0, gene_names=env.hvg_gene_names, signature_genes=env.sig_genes)
    )
    return V, scores


COMPARISONS = [
    # (label, left, right, group)
    ("A->DMSO vs A", "A_then_DMSO", "A", "null"),
    ("B->DMSO vs B", "B_then_DMSO", "B", "null"),
    ("DMSO->A vs A", "DMSO_then_A", "A", "null"),
    ("DMSO->B vs B", "DMSO_then_B", "B", "null"),
    ("DMSO->DMSO vs DMSO", "DMSO_then_DMSO", "DMSO", "null"),
    ("A->B vs B->A", "A_then_B", "B_then_A", "order"),
    ("A->B vs additive", "A_then_B", "additive", "additivity"),
    ("B->A vs additive", "B_then_A", "additive", "additivity"),
    ("co_mean vs additive", "co_mean", "additive", "additivity"),
    ("co_sum vs additive", "co_sum", "additive", "additivity"),
    ("co_mean vs A->B", "co_mean", "A_then_B", "co_vs_seq"),
    ("co_mean vs B->A", "co_mean", "B_then_A", "co_vs_seq"),
    ("co_sum vs co_mean", "co_sum", "co_mean", "co_vs_seq"),
    ("A vs B", "A", "B", "reference"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drug-a", default="palbociclib")
    ap.add_argument("--drug-b", default="venetoclax")
    ap.add_argument("--dose", type=float, default=0.5, help="concentration in uM for both drugs")
    ap.add_argument("--tag", default=None, help="output filename suffix (default: <a>_<b>_<dose>uM)")
    args = ap.parse_args()

    env = RelearnChemicalEnv(EnvConfig(horizon=2))
    key_a = resolve_drug(env, args.drug_a, args.dose)
    key_b = resolve_drug(env, args.drug_b, args.dose)
    dmso_key = env.cfg.dmso_control_pert

    print(f"A = {key_a}\nB = {key_b}\nDMSO = {dmso_key}\n")

    V, scores = run_arms(env, key_a, key_b, dmso_key)

    # per-arm summary
    arm_rows = [
        {
            "arm": k,
            "displacement_norm": float(np.linalg.norm(v)),
            "ucell_score": scores.get(k, float("nan")),
            "cos_to_A": cos(v, V["A"]),
            "cos_to_B": cos(v, V["B"]),
        }
        for k, v in V.items()
    ]

    cmp_rows = [
        {
            "comparison": label,
            "group": group,
            "cosine": cos(V[l], V[r]),
            "rel_residual": rel_residual(V[l], V[r]),
            "norm_left": float(np.linalg.norm(V[l])),
            "norm_right": float(np.linalg.norm(V[r])),
        }
        for label, l, r, group in COMPARISONS
    ]

    tag = args.tag or f"{args.drug_a}_{args.drug_b}_{args.dose}uM".replace(" ", "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"order_additivity_arms_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arm_rows[0].keys()))
        w.writeheader()
        w.writerows(arm_rows)
    with open(OUT_DIR / f"order_additivity_comparisons_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader()
        w.writerows(cmp_rows)
    np.savez_compressed(
        OUT_DIR / f"order_additivity_vectors_{tag}.npz",
        gene_names=env.hvg_gene_names,
        **{k: v.astype(np.float32) for k, v in V.items()},
    )

    # ---- report ----
    print(f"baseline UCell = {scores['baseline']:.4f}\n")
    print(f"{'arm':<18}{'|v|':>10}{'UCell':>10}{'cos->A':>10}{'cos->B':>10}")
    for r in arm_rows:
        print(f"{r['arm']:<18}{r['displacement_norm']:>10.4f}{r['ucell_score']:>10.4f}"
              f"{r['cos_to_A']:>10.4f}{r['cos_to_B']:>10.4f}")

    print(f"\n{'comparison':<24}{'cosine':>10}{'rel_resid':>12}   group")
    for r in cmp_rows:
        print(f"{r['comparison']:<24}{r['cosine']:>10.4f}{r['rel_residual']:>12.4f}   {r['group']}")

    floor = max(r["rel_residual"] for r in cmp_rows if r["group"] == "null")
    order = next(r for r in cmp_rows if r["comparison"] == "A->B vs B->A")
    print(f"\n--- verdicts ---")
    print(f"drift floor (worst wash-out null):   rel_resid = {floor:.4f}")
    print(f"order effect (A->B vs B->A):         rel_resid = {order['rel_residual']:.4f} "
          f"(cos={order['cosine']:.4f})")
    ratio = order["rel_residual"] / floor if floor > 0 else float("inf")
    print(f"order / floor ratio:                 {ratio:.2f}x  ->  "
          f"{'ORDER MATTERS' if ratio > 2 else 'NOT DISTINGUISHABLE FROM DRIFT'}")

    for label in ("A->B vs additive", "co_mean vs additive"):
        r = next(c for c in cmp_rows if c["comparison"] == label)
        rr = r["rel_residual"] / floor if floor > 0 else float("inf")
        print(f"{label:<36} rel_resid = {r['rel_residual']:.4f}  ({rr:.2f}x floor)  "
              f"-> {'NON-ADDITIVE' if rr > 2 else 'additive within drift'}")

    # genes driving the order effect
    d = V["A_then_B"] - V["B_then_A"]
    top = np.argsort(-np.abs(d))[:15]
    print("\ntop genes by |A->B  -  B->A|:")
    for i in top:
        print(f"  {env.hvg_gene_names[i]:<12} delta={d[i]:+.4f}  "
              f"A->B={V['A_then_B'][i]:+.4f}  B->A={V['B_then_A'][i]:+.4f}")

    print(json.dumps({"floor": floor, "order_rel_resid": order["rel_residual"],
                      "order_over_floor": ratio}, indent=2))


if __name__ == "__main__":
    main()
