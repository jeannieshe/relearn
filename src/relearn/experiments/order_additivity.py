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

The per-arm "score" column is whatever cfg.reward_fn computes (see
rewards.py) -- default "ucell" (mean per-cell UCell-vs-apoptosis-signature
score); pass --reward-fn edistance_from_control to report E-distance from a
fixed real-DMSO reference cloud instead:
    python src/relearn/experiments/order_additivity.py \
        --drug-a palbociclib --drug-b venetoclax --dose 0.5 \
        --reward-fn edistance_from_control --reward-seed 0
Unlike the displacement/cosine/additivity analysis (which is always computed
on pseudobulk, mean-over-cells vectors -- see run_arms()), the score column
is computed on each arm's FULL predicted cell-state set, since a
distributional metric like E-distance needs a real cloud on each side, not
a single already-averaged point.

GAIN METRICS (input_change_x_floor / output_change_x_floor / gain_out_per_in),
same definitions as basal_control_sweep.py: every two-hop arm here (A_then_B,
B_then_A, ...) feeds STATE's own predicted intermediate state into the second
forward pass instead of a real cell measurement -- exactly the basal
substitution basal_control_sweep.py sweeps deliberately, arising here as a
side effect of chaining two passes. Each two-hop arm is compared against its
single-hop counterpart with the same final action (A_then_B vs B, DMSO_then_A
vs A, ...): input_change_x_floor is how far the substituted basal (the
predicted intermediate state) sits from a real DMSO draw, and
output_change_x_floor is how far the resulting output sits from that
counterpart's output -- both normalized by a real split-half floor (an
independent second real DMSO draw, run through the same final action).
Single-hop arms (A, B, DMSO, co_mean, co_sum) have no basal substitution to
measure, so these columns are NaN for them, matching how basal_control_sweep.py
never reports a self-vs-self row for its own reference condition either.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from relearn.config import EnvConfig
from relearn.envs.small_molecules import RelearnChemicalEnv
from relearn.rewards import energy_distance

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

    # the real num_cells-sized starting set (shape [S, D]) -- not a single
    # averaged vector. Each arm below runs the full set through STATE (which
    # attends across the S cells together, see state_model.py) and returns
    # the full predicted [S, D] gene-space set; pseudobulking (mean over
    # cells) happens below, separately for the displacement analysis.
    x0_state = env._initial_cell_set
    x0_full = np.asarray(env._to_gene_expression(x0_state), dtype=np.float64)
    x0 = x0_full.mean(axis=0)

    def seq(*action_idxs):
        """
        Apply actions in order from the fixed baseline. Returns
        (raw_state_fed_into_the_last_action, full_[S,D]_gene_expr_after_it) --
        the raw (un-decoded, embed_key-space) state before the last action is
        exactly the "basal" that a real measurement would occupy for a
        single-hop arm sharing the same final action; see the module
        docstring's GAIN METRICS note.
        """
        s = x0_state
        for a in action_idxs[:-1]:
            s = tm.step(s, a)
        raw_before_last = s
        s = tm.step(s, action_idxs[-1])
        return raw_before_last, np.asarray(env._to_gene_expression(s), dtype=np.float64)

    def combo(weight: float):
        """One step with a combined perturbation vector: weight*(onehot_A + onehot_B).
        Returns (x0_state, full_[S,D]_gene_expr) -- always single-hop, so the
        "before" state is just x0_state itself (see reference_output_key below,
        combo arms are their own reference and this is never actually used)."""
        pv = weight * (tm.pert_map[key_a].float() + tm.pert_map[key_b].float())  # type: ignore[attr-defined]
        s = tm.step_with_pert_vector(x0_state, pv)  # type: ignore[attr-defined]
        return x0_state, np.asarray(env._to_gene_expression(s), dtype=np.float64)

    raw_before: dict[str, np.ndarray] = {}
    arms_full: dict[str, np.ndarray] = {}
    for key, (before, full) in {
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
    }.items():
        raw_before[key], arms_full[key] = before, full

    # pseudobulk (mean over cells) per arm -- what the displacement / cosine /
    # additivity analysis below operates on, since those need single vectors.
    arms = {k: full.mean(axis=0) for k, full in arms_full.items()}

    # displacements from the shared baseline
    V = {k: expr - x0 for k, expr in arms.items()}
    V["additive"] = V["A"] + V["B"]

    # scores go through whatever reward function cfg.reward_fn selects (see
    # rewards.py), on each arm's FULL predicted cell set -- not the
    # pseudobulk mean, since a distributional metric like E-distance needs a
    # real cloud on each side (collapsing to one point would silently zero
    # out that arm's own spread and stop measuring anything meaningful).
    scores = {k: env._score(full) for k, full in arms_full.items()}
    scores["baseline"] = env._score(x0_full)

    # ---- GAIN METRICS (basal_control_sweep.py-style) ----
    # each two-hop arm's single-hop counterpart with the same final action --
    # e.g. A_then_B's "real basal, same final action" comparison point is B.
    # Single-hop arms map to themselves (no substitution occurred).
    reference_output_key = {
        "A": "A", "B": "B", "A_then_B": "B", "B_then_A": "A",
        "co_mean": "co_mean", "co_sum": "co_sum",
        "A_then_DMSO": "DMSO", "B_then_DMSO": "DMSO",
        "DMSO_then_A": "A", "DMSO_then_B": "B",
        "DMSO": "DMSO", "DMSO_then_DMSO": "DMSO",
    }

    # split-half floor: an independent second real DMSO draw from the same
    # pool env's own starting set came from (see _load_dmso_control_pool()).
    x0_state_b = env._draw_cell_set()
    floor_in = energy_distance(
        np.asarray(x0_state_b, dtype=np.float64), np.asarray(x0_state, dtype=np.float64)
    )

    # output floor per unique reference arm: apply that SAME final action to
    # x0_state_b instead of x0_state, and compare to the (already-computed)
    # real-basal output -- the noise floor for "how much does this
    # perturbation's predicted output move, given only real sampling noise
    # on the input side."
    floor_out: dict[str, float] = {}
    for ref_key in set(reference_output_key.values()):
        if ref_key in ("co_mean", "co_sum"):
            weight = 0.5 if ref_key == "co_mean" else 1.0
            pv = weight * (tm.pert_map[key_a].float() + tm.pert_map[key_b].float())  # type: ignore[attr-defined]
            s_b = tm.step_with_pert_vector(x0_state_b, pv)  # type: ignore[attr-defined]
        else:
            idx = {"A": idx_a, "B": idx_b, "DMSO": idx_dmso}[ref_key]
            s_b = tm.step(x0_state_b, idx)
        out_b = np.asarray(env._to_gene_expression(s_b), dtype=np.float64)
        floor_out[ref_key] = energy_distance(out_b, arms_full[ref_key])

    gain = {}
    for k in arms_full:
        ref_key = reference_output_key[k]
        cos_ref = cos(V[k], V[ref_key])
        if k == ref_key:
            # single-hop arm -- it IS the reference, no basal was substituted
            gain[k] = {"input_change_x_floor": float("nan"), "output_change_x_floor": float("nan"),
                       "gain_out_per_in": float("nan"), "cos_vs_reference_output": cos_ref}
            continue
        in_dist = energy_distance(
            np.asarray(raw_before[k], dtype=np.float64), np.asarray(x0_state, dtype=np.float64)
        )
        in_x = in_dist / floor_in if floor_in > 0 else float("nan")
        out_dist = energy_distance(arms_full[k], arms_full[ref_key])
        out_x = out_dist / floor_out[ref_key] if floor_out[ref_key] > 0 else float("nan")
        gain[k] = {
            "input_change_x_floor": in_x,
            "output_change_x_floor": out_x,
            "gain_out_per_in": out_x / in_x if in_x and in_x > 0 else float("nan"),
            "cos_vs_reference_output": cos_ref,
        }
    gain["additive"] = {"input_change_x_floor": float("nan"), "output_change_x_floor": float("nan"),
                         "gain_out_per_in": float("nan"), "cos_vs_reference_output": float("nan")}

    return V, scores, gain


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
    ap.add_argument("--n-cells", type=int, default=256,
                    help="size of the real starting cell-sentence (matches the checkpoint's "
                         "trained cell_sentence_len; S=1 is uninformative -- see the "
                         "2026-07-25 EXPERIMENTS.md entry, single cells are 96%% zeros)")
    ap.add_argument("--seed", type=int, default=None, help="seed for which real DMSO cells are drawn")
    ap.add_argument("--reward-fn", default="ucell", choices=["ucell", "edistance_from_control"],
                    help="scoring strategy reported per arm (see rewards.py); edistance_from_control "
                         "measures each arm's full predicted cell cloud against one fixed reference "
                         "draw from the real DMSO control pool, instead of the UCell apoptosis score")
    ap.add_argument("--reward-reference-n-cells", type=int, default=256,
                    help="reference cloud size, only used by --reward-fn edistance_from_control")
    ap.add_argument("--reward-seed", type=int, default=None,
                    help="seed for which reference cells edistance_from_control draws -- fixed for "
                         "the whole run so every arm is compared against the same reference cloud")
    ap.add_argument("--tag", default=None, help="output filename suffix (default: <a>_<b>_<dose>uM)")
    args = ap.parse_args()

    env = RelearnChemicalEnv(
        EnvConfig(
            horizon=2, num_cells=args.n_cells, reward_fn=args.reward_fn,
            reward_reference_n_cells=args.reward_reference_n_cells, reward_seed=args.reward_seed,
        ),
        seed=args.seed,
    )
    key_a = resolve_drug(env, args.drug_a, args.dose)
    key_b = resolve_drug(env, args.drug_b, args.dose)
    dmso_key = env.cfg.dmso_control_pert

    print(f"A = {key_a}\nB = {key_b}\nDMSO = {dmso_key}\n")

    V, scores, gain = run_arms(env, key_a, key_b, dmso_key)

    # per-arm summary
    arm_rows = [
        {
            "arm": k,
            "displacement_norm": float(np.linalg.norm(v)),
            "score": scores.get(k, float("nan")),
            "cos_to_A": cos(v, V["A"]),
            "cos_to_B": cos(v, V["B"]),
            **gain[k],
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

    tag = args.tag or (
        f"{args.drug_a}_{args.drug_b}_{args.dose}uM".replace(" ", "")
        + ("" if args.reward_fn == "ucell" else f"_{args.reward_fn}")
    )
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
    print(f"baseline score ({args.reward_fn}) = {scores['baseline']:.4f}\n")
    print(f"{'arm':<18}{'|v|':>10}{'score':>10}{'cos->A':>10}{'cos->B':>10}"
          f"{'in x floor':>12}{'out x floor':>13}{'gain':>8}")
    for r in arm_rows:
        print(f"{r['arm']:<18}{r['displacement_norm']:>10.4f}{r['score']:>10.4f}"
              f"{r['cos_to_A']:>10.4f}{r['cos_to_B']:>10.4f}"
              f"{r['input_change_x_floor']:>12.4f}{r['output_change_x_floor']:>13.4f}"
              f"{r['gain_out_per_in']:>8.3f}")

    print("\n(in/out x floor and gain are basal_control_sweep.py-style GAIN metrics -- NaN for "
          "single-hop arms A/B/DMSO/co_mean/co_sum, which have no substituted basal to measure. "
          "For two-hop arms, 'in x floor' is how far STATE's own predicted intermediate state "
          "sits from a real DMSO draw, and 'out x floor' is how far the resulting output sits "
          "from the single-hop counterpart with the same final action -- both in units of a real "
          "split-half noise floor. gain = out / in: >1 means the model amplifies the basal "
          "substitution past what real sampling noise alone would produce.)")

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
