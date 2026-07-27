"""
Geometry of perturbation effects in gene-expression space, as a replacement for
the scalar apoptosis reward.

The reward collapses a 2000-dim predicted expression change down to one number,
so two drugs with identical UCell scores are indistinguishable even if they move
the cell in opposite directions. This script keeps the vector.

Everything here is computed on *displacement* vectors

    v_d = expr(after drug d) - expr(baseline)

not on the raw post-perturbation states. That distinction is the whole ballgame:
raw states are dominated by the shared baseline profile, so cosine(x_a, x_b)
between any two post-drug states sits at ~0.999 regardless of what the drugs did.
Subtracting the common baseline is what makes angles informative.

Scored in the 2000-HVG gene basis via env._to_gene_expression(), so X_hvg and
X_state runs are directly comparable (the SE latent is decoded back to genes
first; this is fair because presumably whatever reward we use will also rely on
the SE output being first decoded into the 2000 HVG space).

Run with: python src/relearn/experiments/expression_geometry.py [env=sw480_se]
"""

import csv
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from relearn.envs.small_molecules import RelearnChemicalEnv

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def collect_displacements(env: RelearnChemicalEnv):
    """
    Apply each of the 1138 drugs once to the fixed baseline state and return the
    resulting displacement vectors in 2000-HVG gene space.

    Deterministic for the same reason the other sweeps are (see
    enumerate_perturbations.py): reset() returns a fixed state and STATE's
    forward pass has no dropout at eval, so one pass per drug is exhaustive.
    """
    env.reset()
    x0 = np.asarray(env._to_gene_expression(env.initial_cell_state), dtype=np.float64)

    n_actions = env.action_space.n
    V = np.empty((n_actions, x0.shape[0]), dtype=np.float64)
    scores = np.empty(n_actions, dtype=np.float64)

    for action in range(n_actions):
        env.reset()
        obs, _, _, _, info = env.step(action)
        V[action] = np.asarray(env._to_gene_expression(obs), dtype=np.float64) - x0
        # keep the reward around purely so geometry can be correlated against it
        scores[action] = info["apoptosis score"]

    return x0, V, scores


def unit_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize, leaving zero rows as zero instead of NaN."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(norms, eps)


def signature_axis(env: RelearnChemicalEnv) -> tuple[np.ndarray, int]:
    """
    Unit vector pointing along "raise the apoptosis signature genes".

    This is the geometric stand-in for the reward, and unlike UCell it is
    differentiable-friendly and sign-aware: a drug that *suppresses* the
    signature gets a negative projection, where UCell would just give it a
    middling score. Built as the normalized indicator over the signature genes
    present in STATE's 2000-HVG panel.
    """
    mask = np.isin(env.hvg_gene_names, np.asarray(env.sig_genes, dtype=str))
    n_present = int(mask.sum())
    if n_present == 0:
        raise ValueError("no signature genes overlap the HVG panel -- check gmt_path / hvg_gene_names_path")
    axis = mask.astype(np.float64)
    return axis / np.linalg.norm(axis), n_present


def effective_rank(V: np.ndarray) -> tuple[float, int, np.ndarray]:
    """
    How many independent directions the model actually moves the cell along.

    Returns the participation ratio (a soft, continuous count of dominant
    dimensions), the hard count of components needed for 95% of the variance,
    and the fraction of variance in each of the top singular directions.

    A tiny effective rank means the reward landscape *has* to be flat: if every
    drug's effect lands in a 2-3 dimensional subspace, there is no room for
    1138 meaningfully different outcomes.
    """
    s = np.linalg.svd(V, compute_uv=False)
    var = s**2
    total = var.sum()
    if total == 0:
        return 0.0, 0, var
    pr = float(total**2 / (var**2).sum())          # participation ratio
    frac = var / total
    n95 = int(np.searchsorted(np.cumsum(frac), 0.95) + 1)
    return pr, n95, frac


def summarize(env, x0, V, scores):
    """Per-drug geometry + the population-level diagnostics."""
    axis, n_sig_present = signature_axis(env)

    norms = np.linalg.norm(V, axis=1)
    U = unit_rows(V)

    # the common response axis: the average direction every drug pushes in.
    # cos(v_d, mean) near 1.0 for everything => the model has essentially one
    # response mode and drug identity barely matters.
    mean_dir = V.mean(axis=0)
    mean_dir_unit = mean_dir / max(np.linalg.norm(mean_dir), 1e-12)
    cos_to_mean = U @ mean_dir_unit

    # signed on-target component, and what's left over
    on_target = V @ axis
    off_target = np.linalg.norm(V - np.outer(on_target, axis), axis=1)
    specificity = on_target / np.maximum(norms, 1e-12)   # == cos(v_d, apoptosis axis)

    # drug x drug direction similarity. computed on unit rows so it's pure angle.
    cosmat = U @ U.T

    per_drug = []
    for i in range(V.shape[0]):
        per_drug.append({
            "action": i,
            "drug": env.drug_list[i],
            "displacement_norm": float(norms[i]),
            "cos_to_mean_response": float(cos_to_mean[i]),
            "on_target_projection": float(on_target[i]),
            "off_target_norm": float(off_target[i]),
            "specificity": float(specificity[i]),
            "ucell_score": float(scores[i]),
        })

    pr_raw, n95_raw, frac_raw = effective_rank(V)
    pr_ctr, n95_ctr, frac_ctr = effective_rank(V - V.mean(axis=0))

    # off-diagonal pairwise cosines only
    iu = np.triu_indices_from(cosmat, k=1)
    pairwise = cosmat[iu]

    stats = {
        "n_drugs": V.shape[0],
        "n_genes": V.shape[1],
        "n_signature_genes_in_panel": n_sig_present,
        "norm_mean": float(norms.mean()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "cos_to_mean_response_mean": float(cos_to_mean.mean()),
        "cos_to_mean_response_min": float(cos_to_mean.min()),
        "pairwise_cos_mean": float(pairwise.mean()),
        "pairwise_cos_p05": float(np.quantile(pairwise, 0.05)),
        "pairwise_cos_p95": float(np.quantile(pairwise, 0.95)),
        "specificity_mean": float(specificity.mean()),
        "specificity_max": float(specificity.max()),
        "participation_ratio_raw": pr_raw,
        "n_pcs_95pct_raw": n95_raw,
        "participation_ratio_centered": pr_ctr,
        "n_pcs_95pct_centered": n95_ctr,
        "top5_variance_fraction_raw": frac_raw[:5].tolist(),
        "top5_variance_fraction_centered": frac_ctr[:5].tolist(),
        "spearman_specificity_vs_ucell": float(
            np.corrcoef(
                np.argsort(np.argsort(specificity)),
                np.argsort(np.argsort(scores)),
            )[0, 1]
        ),
    }
    return per_drug, cosmat, stats


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    env = RelearnChemicalEnv(cfg.env)

    out_dir = REPO_ROOT / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = env.embed_key.removeprefix("X_")

    x0, V, scores = collect_displacements(env)
    per_drug, cosmat, stats = summarize(env, x0, V, scores)

    # the raw vectors, so notebook-side work (UMAP, clustering, MoA grouping)
    # never has to re-run the 1138-pass sweep
    np.savez_compressed(
        out_dir / f"displacements_{tag}.npz",
        displacements=V.astype(np.float32),
        baseline=x0.astype(np.float32),
        ucell_scores=scores,
        gene_names=env.hvg_gene_names,
        drugs=np.array([str(d) for d in env.drug_list], dtype=object),
    )

    csv_path = out_dir / f"expression_geometry_{tag}.csv"
    fieldnames = list(per_drug[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(per_drug, key=lambda r: r["specificity"], reverse=True))

    print(f"\nwrote {csv_path}")
    print(f"wrote {out_dir / f'displacements_{tag}.npz'}")
    print(f"\n--- geometry of {stats['n_drugs']} drugs in {stats['n_genes']}-gene space "
          f"(embed_key={env.embed_key}) ---")
    print(f"signature genes present in panel: {stats['n_signature_genes_in_panel']}")
    print(f"displacement norm:      mean={stats['norm_mean']:.4f} "
          f"min={stats['norm_min']:.4f} max={stats['norm_max']:.4f}")
    print(f"cos to mean response:   mean={stats['cos_to_mean_response_mean']:.4f} "
          f"min={stats['cos_to_mean_response_min']:.4f}   (near 1.0 => one shared response mode)")
    print(f"pairwise drug-drug cos: mean={stats['pairwise_cos_mean']:.4f} "
          f"p05={stats['pairwise_cos_p05']:.4f} p95={stats['pairwise_cos_p95']:.4f}")
    print(f"specificity (cos to apoptosis axis): mean={stats['specificity_mean']:.4f} "
          f"max={stats['specificity_max']:.4f}")
    print(f"effective dim (raw):      participation_ratio={stats['participation_ratio_raw']:.2f} "
          f"n_pcs_95%={stats['n_pcs_95pct_raw']}")
    print(f"effective dim (centered): participation_ratio={stats['participation_ratio_centered']:.2f} "
          f"n_pcs_95%={stats['n_pcs_95pct_centered']}")
    print(f"top-5 variance fraction (centered): "
          f"{[round(x, 4) for x in stats['top5_variance_fraction_centered']]}")
    print(f"rank corr specificity vs. UCell reward: {stats['spearman_specificity_vs_ucell']:.4f}")

    print("\nTop 10 by specificity (most on-axis toward apoptosis):")
    for r in sorted(per_drug, key=lambda r: r["specificity"], reverse=True)[:10]:
        print(f"  {r['drug']}: spec={r['specificity']:+.4f} "
              f"|v|={r['displacement_norm']:.4f} ucell={r['ucell_score']:.4f}")

    print("\nTop 10 by displacement magnitude (biggest movers, on-axis or not):")
    for r in sorted(per_drug, key=lambda r: r["displacement_norm"], reverse=True)[:10]:
        print(f"  {r['drug']}: |v|={r['displacement_norm']:.4f} "
              f"spec={r['specificity']:+.4f} ucell={r['ucell_score']:.4f}")

    print("\nMost distinctive directions (lowest cos to the mean response):")
    for r in sorted(per_drug, key=lambda r: r["cos_to_mean_response"])[:10]:
        print(f"  {r['drug']}: cos_to_mean={r['cos_to_mean_response']:+.4f} "
              f"|v|={r['displacement_norm']:.4f} ucell={r['ucell_score']:.4f}")


if __name__ == "__main__":
    main()