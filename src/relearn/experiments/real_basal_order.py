"""
Order / additivity for a drug pair, using Tahoe's REAL measured cells as basal
instead of iterating STATE on its own predictions.

Motivation: order_additivity.py feeds STATE's predicted A-state back in as
ctrl_cell_emb for step 2. That input is a dense ReLU output, nothing like a real
cell, and the resulting drift (rel_resid ~1.0 on a DMSO wash-out that should be
a no-op) swamps every effect we wanted to measure. Here step 1 is replaced by
measurement: to ask "what does B do after A", we hand the model the real
palbociclib-treated SW480 cells from Tahoe.

Two further fixes come along with that:

  * cell_sentence_len = 256. The backbone is a *bidirectional* transformer over
    a sentence of cells, but every call in this repo passes S=1, so each cell
    attends only to itself -- off-distribution for the transformer regardless of
    what the basal is. Real populations let us run sentences at the trained
    length. --sentence-len 1 reruns everything at S=1 as a sensitivity check.

  * A real noise floor. Two disjoint 256-cell draws from the *same* real
    population give the distance you get from measurement + biological
    variability alone. Any order or non-additivity effect has to clear that bar.
    This replaces the synthetic drift floor, which was not a biological scale.

WHAT THIS STILL DOES NOT DO. ST is trained as a clean control -> perturbed map;
`state tx infer` samples basal only from the DMSO control pool (see the notebook
notes in notebooks/alaysia/state_embedding_analysis.ipynb). Handing it
A-treated cells as basal is milder extrapolation than a synthetic vector, but it
is still extrapolation -- the model cannot know the basal already contains A, so
it applies B's control-relative effect on top. And Tahoe is single-agent, so
there is no measured A+B to check against: this quantifies DRIFT, not the
ACCURACY of the combination prediction. Tight error bars here do not mean the
combination is right.

Run with:
    python src/relearn/experiments/real_basal_order.py \
        --drug-a palbociclib --drug-b venetoclax --dose 0.5 --n-draws 20
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from relearn.config import EnvConfig
from relearn.rewards import energy_distance
from relearn.transitions.state_model import StateTransitionModel
from relearn.utils import _load_gmt_signature, ucell_score

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "artifacts"


# --------------------------------------------------------------------------
# real-cell loading
# --------------------------------------------------------------------------

def find_cell_line_file(tahoe_se_dir: Path, accession: str) -> Path:
    for candidate in sorted(tahoe_se_dir.glob("c*.h5ad")):
        with h5py.File(candidate, "r") as f:
            cl = f["obs"]["cell_line"]["categories"][0]
            cl = cl.decode() if isinstance(cl, bytes) else cl
            if cl == accession:
                return candidate
    raise FileNotFoundError(f"no Tahoe h5ad under {tahoe_se_dir} for cell line {accession}")


def load_pools(h5_path: Path, embed_key: str, labels: list[str], max_cells: int, rng) -> dict:
    """
    Read the real measured cells for each requested perturbation label.

    Subsamples to max_cells per label -- the DMSO control pool is ~148k cells in
    SW480 and we only ever draw sentences of a few hundred from it.
    """
    pools = {}
    with h5py.File(h5_path, "r") as f:
        cats = [c.decode() if isinstance(c, bytes) else c for c in f["obs"]["drugname_drugconc"]["categories"][:]]
        codes = f["obs"]["drugname_drugconc"]["codes"][:]
        dset = f["obsm"][embed_key]
        for label in labels:
            if label not in cats:
                raise KeyError(f"{label!r} not present in {h5_path.name}")
            rows = np.where(codes == cats.index(label))[0]
            if len(rows) > max_cells:
                rows = rng.choice(rows, size=max_cells, replace=False)
            rows = np.sort(rows)  # h5py fancy indexing needs sorted, unique
            pools[label] = dset[rows, :].astype(np.float32)
    return pools


def resolve_label(cats: list[str], name: str, dose: float) -> str:
    matches = [c for c in cats if name.lower() in c.lower() and f", {dose}," in c]
    if len(matches) != 1:
        raise KeyError(f"{name!r} @ {dose}uM matched {len(matches)} labels: {matches[:5]}")
    return matches[0]


# --------------------------------------------------------------------------
# model calls
# --------------------------------------------------------------------------

def predict(tm: StateTransitionModel, basal: np.ndarray, pert_vec: torch.Tensor) -> np.ndarray:
    """
    One forward pass over a sentence of S cells sharing a perturbation.

    forward(padded=False) reshapes the batch to [1, S, D], i.e. the S cells form
    a single sentence and attend to each other -- which is why S should be
    cell_sentence_len, not 1.
    """
    S = basal.shape[0]
    batch = {
        "ctrl_cell_emb": torch.tensor(basal, dtype=torch.float32, device=tm._device),
        "pert_emb": pert_vec.float().unsqueeze(0).repeat(S, 1).to(tm._device),
        "pert_name": ["<sentence>"] * S,
    }
    with torch.no_grad():
        pred = tm._model.forward(batch, padded=False)
    return pred.reshape(S, -1).cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------
# distribution-level metrics
# --------------------------------------------------------------------------
# energy_distance() moved to relearn.rewards (imported above) so
# EDistanceFromControlReward and this script share one implementation.


def cos(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / denom) if denom > 0 else float("nan")


def rel_residual(u: np.ndarray, v: np.ndarray, ref_scale: float) -> float:
    """
    ||u - v|| expressed as a fraction of a FIXED reference scale -- the typical
    real single-drug effect size -- rather than of the pair's own magnitudes.

    Normalizing by the pair's own norms (the obvious choice) is wrong for the
    split-half floor: two half-means differ by roughly their own sampling noise,
    so that ratio pins to ~1.0 by construction no matter how tight the data is,
    and every real effect then looks small against it. A fixed denominator keeps
    the floor, the order effect, and the additivity residuals commensurate.
    """
    return float(np.linalg.norm(u - v) / ref_scale) if ref_scale > 0 else float("nan")


# --------------------------------------------------------------------------
# one bootstrap draw
# --------------------------------------------------------------------------

COMPARISONS = [
    # label, left arm, right arm, group
    ("ctrl->A vs real_A", "ctrl_to_A", "real_A", "pass1_accuracy"),
    ("ctrl->B vs real_B", "ctrl_to_B", "real_B", "pass1_accuracy"),
    ("ctrl->DMSO vs real_DMSO", "ctrl_to_DMSO", "real_DMSO", "fixed_point"),
    ("realA->DMSO vs real_A", "realA_to_DMSO", "real_A", "washout_null"),
    ("realB->DMSO vs real_B", "realB_to_DMSO", "real_B", "washout_null"),
    ("DMSO split-half", "dmso_half1", "dmso_half2", "noise_floor"),
    ("real_A split-half", "realA_half1", "realA_half2", "noise_floor"),
    # does the basal input matter at all? if realA->B lands on top of ctrl->B,
    # the model ignored the A-treated basal and the "sequence" is just drug B.
    ("realA->B vs ctrl->B", "realA_to_B", "ctrl_to_B", "basal_sensitivity"),
    ("realB->A vs ctrl->A", "realB_to_A", "ctrl_to_A", "basal_sensitivity"),
    ("realA->DMSO vs ctrl->DMSO", "realA_to_DMSO", "ctrl_to_DMSO", "basal_sensitivity"),
    # the same claim measured directly against REAL cells rather than against the
    # model's own single-drug prediction: is "A then B" just B? Stated on its own
    # so it doesn't rest on chaining basal_sensitivity through pass1_accuracy.
    ("realA->B vs real_B", "realA_to_B", "real_B", "seq_vs_real_single"),
    ("realB->A vs real_A", "realB_to_A", "real_A", "seq_vs_real_single"),
    ("realA->B vs realB->A", "realA_to_B", "realB_to_A", "order"),
    ("realA->B vs additive", "realA_to_B", "additive", "additivity"),
    ("realB->A vs additive", "realB_to_A", "additive", "additivity"),
    ("co_mean vs additive", "co_mean", "additive", "additivity"),
    ("co_mean vs realA->B", "co_mean", "realA_to_B", "co_vs_seq"),
    ("co_mean vs realB->A", "co_mean", "realB_to_A", "co_vs_seq"),
    ("real_A vs real_B", "real_A", "real_B", "reference"),
]


def one_draw(tm, pools, label_a, label_b, label_dmso, S, rng, gene_names, sig_genes, x0):
    """Sample fresh sentences from each real pool and run every arm once."""
    def draw(label, n=S):
        pool = pools[label]
        idx = rng.choice(len(pool), size=n, replace=len(pool) < n)
        return pool[idx].astype(np.float64)

    pv_a = tm.pert_map[label_a].float()
    pv_b = tm.pert_map[label_b].float()
    pv_dmso = tm.pert_map[label_dmso].float()

    # real measured populations
    real_dmso = draw(label_dmso)
    real_a = draw(label_a)
    real_b = draw(label_b)

    # disjoint split-halves of the same real pools -> the noise floor
    dmso_2S = draw(label_dmso, 2 * S)
    a_2S = draw(label_a, 2 * S)

    basal_ctrl = draw(label_dmso)          # independent control basal for the ctrl-> arms
    basal_a = draw(label_a)                # real A-treated cells as basal for B
    basal_b = draw(label_b)

    P = {
        "real_DMSO": real_dmso,
        "real_A": real_a,
        "real_B": real_b,
        "dmso_half1": dmso_2S[:S],
        "dmso_half2": dmso_2S[S:],
        "realA_half1": a_2S[:S],
        "realA_half2": a_2S[S:],
        # control -> single drug: the in-distribution, as-trained call
        "ctrl_to_A": predict(tm, basal_ctrl, pv_a),
        "ctrl_to_B": predict(tm, basal_ctrl, pv_b),
        "ctrl_to_DMSO": predict(tm, basal_ctrl, pv_dmso),
        # real drug-treated cells as basal
        "realA_to_B": predict(tm, basal_a, pv_b),
        "realB_to_A": predict(tm, basal_b, pv_a),
        "realA_to_DMSO": predict(tm, basal_a, pv_dmso),
        "realB_to_DMSO": predict(tm, basal_b, pv_dmso),
        # simultaneous co-dose, one pass, control basal (in-distribution basal)
        "co_mean": predict(tm, basal_ctrl, 0.5 * (pv_a + pv_b)),
    }

    # every displacement is measured from the same origin -- the full-pool real
    # DMSO mean, fixed across draws -- so arms with different basals stay
    # comparable and the origin contributes no per-draw sampling noise
    V = {k: pop.mean(axis=0) - x0 for k, pop in P.items()}
    # additive null built from the REAL measured single-drug effects. Given a
    # population form too (real A cells shifted by B's measured mean effect) so
    # energy distance is defined against it, not just the mean comparison.
    V["additive"] = V["real_A"] + V["real_B"]
    P["additive"] = real_a + V["real_B"][None, :]

    # fixed reference scale for every normalized distance: the typical real,
    # measured single-drug effect size. Independent of which pair is compared.
    ref_scale = 0.5 * (np.linalg.norm(V["real_A"]) + np.linalg.norm(V["real_B"]))

    arms = {
        k: {
            "displacement_norm": float(np.linalg.norm(V[k])),
            "ucell": (float(np.mean([ucell_score(c, gene_names=gene_names, signature_genes=sig_genes)
                                     for c in P[k][:32]])) if P[k] is not None else float("nan")),
        }
        for k in V
    }

    comps = {}
    for label, l, r, group in COMPARISONS:
        ed = (energy_distance(P[l], P[r]) if P.get(l) is not None and P.get(r) is not None else float("nan"))
        comps[label] = {
            "group": group,
            "cosine": cos(V[l], V[r]),
            "rel_residual": rel_residual(V[l], V[r], ref_scale),
            "energy_distance": ed,
        }
    return arms, comps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drug-a", default="palbociclib")
    ap.add_argument("--drug-b", default="venetoclax")
    ap.add_argument("--dose", type=float, default=0.5)
    ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--sentence-len", type=int, default=256,
                    help="cells per forward pass; 256 matches cell_sentence_len. Use 1 to reproduce the old regime.")
    ap.add_argument("--max-cells", type=int, default=20000, help="cap per perturbation pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = EnvConfig()
    if cfg.embed_key != "X_hvg":
        raise SystemExit(f"embed_key={cfg.embed_key!r}: this script assumes gene-space output. Use X_hvg.")

    rng = np.random.default_rng(args.seed)
    h5_path = find_cell_line_file(Path(cfg.tahoe_se_dir), cfg.cell_type_accession_number)
    with h5py.File(h5_path, "r") as f:
        cats = [c.decode() if isinstance(c, bytes) else c for c in f["obs"]["drugname_drugconc"]["categories"][:]]
    label_a = resolve_label(cats, args.drug_a, args.dose)
    label_b = resolve_label(cats, args.drug_b, args.dose)
    label_dmso = cfg.dmso_control_pert

    print(f"cell line : {cfg.cell_type_name} ({cfg.cell_type_accession_number}) -> {h5_path.name}")
    print(f"A         : {label_a}\nB         : {label_b}\nDMSO      : {label_dmso}")
    print(f"sentence  : S={args.sentence_len}   draws={args.n_draws}\n")

    pools = load_pools(h5_path, cfg.embed_key, [label_a, label_b, label_dmso], args.max_cells, rng)
    for k, v in pools.items():
        print(f"  loaded {v.shape[0]:>6} real cells for {k}")

    tm = StateTransitionModel(cfg)
    gene_names = np.load(cfg.hvg_gene_names_path, allow_pickle=True).astype(str)
    sig_genes = _load_gmt_signature(cfg.gmt_path, cfg.msigdb_gene_set)

    x0 = pools[label_dmso].mean(axis=0).astype(np.float64)

    all_arms, all_comps = [], []
    for i in range(args.n_draws):
        a, c = one_draw(tm, pools, label_a, label_b, label_dmso,
                        args.sentence_len, rng, gene_names, sig_genes, x0)
        all_arms.append(a)
        all_comps.append(c)
        print(f"  draw {i+1}/{args.n_draws} done", end="\r")
    print()

    def agg(dicts, key, field):
        vals = np.array([d[key][field] for d in dicts], dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    arm_rows = []
    for k in all_arms[0]:
        n_mu, n_sd = agg(all_arms, k, "displacement_norm")
        u_mu, u_sd = agg(all_arms, k, "ucell")
        arm_rows.append({"arm": k, "norm_mean": n_mu, "norm_std": n_sd,
                         "ucell_mean": u_mu, "ucell_std": u_sd})

    cmp_rows = []
    for label, _l, _r, group in COMPARISONS:
        c_mu, c_sd = agg(all_comps, label, "cosine")
        r_mu, r_sd = agg(all_comps, label, "rel_residual")
        e_mu, e_sd = agg(all_comps, label, "energy_distance")
        cmp_rows.append({"comparison": label, "group": group,
                         "cosine_mean": c_mu, "cosine_std": c_sd,
                         "rel_residual_mean": r_mu, "rel_residual_std": r_sd,
                         "energy_distance_mean": e_mu, "energy_distance_std": e_sd})

    tag = args.tag or f"{args.drug_a}_{args.drug_b}_{args.dose}uM_S{args.sentence_len}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"real_basal_arms_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arm_rows[0].keys())); w.writeheader(); w.writerows(arm_rows)
    with open(OUT_DIR / f"real_basal_comparisons_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys())); w.writeheader(); w.writerows(cmp_rows)

    # ---- report ----
    print(f"\n{'arm':<16}{'|v|':>12}{'+/-':>8}{'UCell':>10}{'+/-':>8}")
    for r in arm_rows:
        print(f"{r['arm']:<16}{r['norm_mean']:>12.4f}{r['norm_std']:>8.3f}"
              f"{r['ucell_mean']:>10.4f}{r['ucell_std']:>8.3f}")

    print(f"\n{'comparison':<26}{'cos':>8}{'rel_resid':>12}{'+/-':>7}{'energy':>10}{'+/-':>7}  group")
    for r in cmp_rows:
        print(f"{r['comparison']:<26}{r['cosine_mean']:>8.4f}{r['rel_residual_mean']:>12.4f}"
              f"{r['rel_residual_std']:>7.3f}{r['energy_distance_mean']:>10.3f}"
              f"{r['energy_distance_std']:>7.3f}  {r['group']}")

    by = {r["comparison"]: r for r in cmp_rows}
    floor = max(by["DMSO split-half"]["rel_residual_mean"], by["real_A split-half"]["rel_residual_mean"])
    e_floor = max(by["DMSO split-half"]["energy_distance_mean"], by["real_A split-half"]["energy_distance_mean"])
    order = by["realA->B vs realB->A"]

    print("\n--- verdicts (all ratios vs. the real split-half noise floor) ---")
    print(f"noise floor:  rel_resid={floor:.4f}   energy={e_floor:.4f}")
    for label in ("ctrl->A vs real_A", "ctrl->B vs real_B", "ctrl->DMSO vs real_DMSO",
                  "realA->DMSO vs real_A", "realA->B vs ctrl->B",
                  "realA->B vs real_B", "realB->A vs real_A", "realA->B vs realB->A",
                  "realA->B vs additive", "co_mean vs additive"):
        r = by[label]
        ratio = r["rel_residual_mean"] / floor if floor > 0 else float("inf")
        e_ratio = r["energy_distance_mean"] / e_floor if e_floor > 0 else float("inf")
        print(f"  {label:<28} {ratio:>7.2f}x floor (rel_resid)   {e_ratio:>7.2f}x floor (energy)")

    verdict = "ORDER MATTERS" if order["rel_residual_mean"] > 2 * floor else "NOT DISTINGUISHABLE FROM NOISE"
    print(f"\nORDER: {verdict}  "
          f"(rel_resid={order['rel_residual_mean']:.4f} +/- {order['rel_residual_std']:.4f}, "
          f"cos={order['cosine_mean']:.4f}, floor={floor:.4f})")

    summary = {"drug_a": label_a, "drug_b": label_b, "sentence_len": args.sentence_len,
               "n_draws": args.n_draws, "noise_floor_rel_resid": floor,
               "noise_floor_energy": e_floor,
               "order_rel_resid": order["rel_residual_mean"],
               "order_over_floor": order["rel_residual_mean"] / floor if floor else None,
               "verdict": verdict}
    with open(OUT_DIR / f"real_basal_summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote artifacts/real_basal_{{arms,comparisons}}_{tag}.csv and real_basal_summary_{tag}.json")


if __name__ == "__main__":
    main()
