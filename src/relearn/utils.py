"""
UCell-style gene-signature scoring (Andreatta & Carmona, 2021) for a single
cell's expression profile, e.g. HALLMARK_APOPTOSIS.

UCell mechanics:
  1. Rank all genes in the cell by expression, descending (rank 1 = highest).
  2. Cap ranks at `max_rank` (default 1500) -- genes beyond this are all
     treated as tied at max_rank. This makes the score robust to dropout /
     sparse zeros dominating the ranking, and keeps it stable across gene
     panels of different sizes (relevant since STATE outputs 2000 HVGs, not
     the full transcriptome).
  3. Compute a Mann-Whitney-U-derived AUC on the ranks of the signature
     genes vs. background, and map it to a bounded score in ~[0, 1] via
     U_max = n_sig * max_rank.

No external bio-package dependency required -- pure numpy. If you have many
cells to score (a full AnnData / batch of STATE rollouts), use `ucell_score`
directly on a 2D matrix rather than looping cell-by-cell.
"""

from __future__ import annotations

import numpy as np


def _load_gmt_signature(gmt_path: str, signature_name: str) -> list[str]:
    """
    Load a gene set from a MSigDB .gmt file (e.g. h.all.v2024.1.Hs.symbols.gmt,
    downloaded from https://www.gsea-msigdb.org/gsea/msigdb).

    .gmt format per line: <name>\t<description>\t<gene1>\t<gene2>\t...
    """
    with open(gmt_path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if fields[0] == signature_name:
                return fields[2:]
    raise KeyError(f"Signature '{signature_name}' not found in {gmt_path}")


def ucell_score(
    expr: np.ndarray,
    gene_names: list[str] | np.ndarray,
    signature_genes: list[str],
    max_rank: int = 1500,
) -> np.ndarray:
    """
    Compute UCell score(s) for one gene signature.

    Parameters
    ----------
    expr : np.ndarray, shape (n_genes,) or (n_cells, n_genes)
        Expression matrix or single-cell vector. Should already be
        normalized the way STATE's X_hvg is (library-size normalize to
        1e4 + log1p) -- UCell only uses within-cell rank order, so
        monotonic transforms of raw counts don't change the score, but
        log1p vs. raw does change how ties are broken at very low counts.
    gene_names : sequence of str, length n_genes
        Column labels for `expr`, in the same order as STATE's HVG panel.
    signature_genes : list of str
        Gene symbols making up the signature (e.g. HALLMARK_APOPTOSIS).
    max_rank : int, default 1500
        Rank cap. UCell's default; lower it if your panel is much smaller
        than a full transcriptome (STATE's 2000-HVG panel is already close
        to this, so max_rank=1500 or even 2000 is reasonable -- see note
        below).

    Returns
    -------
    np.ndarray, shape () or (n_cells,)
        UCell score(s) in [0, 1]. Higher = signature genes rank higher
        (are more highly expressed) in that cell.
    """
    expr = np.atleast_2d(expr)
    n_cells, n_genes = expr.shape

    gene_names = np.asarray(gene_names)
    if gene_names.shape[0] != n_genes:
        raise ValueError("gene_names length must match expr's gene axis")

    # Effective cap can't exceed the panel size (STATE's HVG panel may be
    # smaller than UCell's transcriptome-scale default of 1500-3000).
    eff_max_rank = min(max_rank, n_genes)

    sig_mask = np.isin(gene_names, signature_genes)
    n_sig = int(sig_mask.sum())
    if n_sig == 0:
        raise ValueError(
            "None of the signature genes were found in gene_names -- "
            "check symbol casing/aliasing between your panel and the .gmt file."
        )

    # Rank genes within each cell, descending expression -> rank 1 = highest.
    # argsort trick: rank of element = position in the sorted-descending order.
    order = np.argsort(-expr, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    row_idx = np.arange(n_cells)[:, None]
    ranks[row_idx, order] = np.arange(1, n_genes + 1)[None, :]

    # Cap ranks at eff_max_rank (all lower-expressed genes tie at the cap).
    ranks = np.minimum(ranks, eff_max_rank)

    sig_ranks = ranks[:, sig_mask]  # (n_cells, n_sig)
    rank_sum = sig_ranks.sum(axis=1)

    # Mann-Whitney U on ranks, normalized to a bounded [0, 1] score.
    u_stat = rank_sum - n_sig * (n_sig + 1) / 2.0
    u_max = n_sig * eff_max_rank
    auc = u_stat / u_max  # in [0, 1], but oriented so *higher* AUC = *lower* expression
    score = 1.0 - auc

    return score[0] if score.shape[0] == 1 else score


if __name__ == "__main__":
    # Smoke test: synthetic panel, signature genes deliberately given high
    # expression -> score should land near 1.0.
    rng = np.random.default_rng(0)
    n_genes = 2000
    genes = np.array([f"GENE{i}" for i in range(n_genes)])
    sig = list(genes[:161])  # pretend these 161 are HALLMARK_APOPTOSIS

    expr_high = rng.normal(1.0, 0.3, size=n_genes)
    expr_high[:161] += 5.0  # boost signature genes' expression
    expr_low = rng.normal(1.0, 0.3, size=n_genes)
    expr_low[:161] -= 1.0  # suppress signature genes

    s_high = ucell_score(expr_high, genes, sig)
    s_low = ucell_score(expr_low, genes, sig)
    print(f"score when signature genes are up-expressed:   {s_high:.4f}")
    print(f"score when signature genes are down-expressed: {s_low:.4f}")
    assert s_high > 0.9, "expected near-1.0 score for strongly up-expressed signature"
    assert s_low < 0.5, "expected low score for down-expressed signature"

    # Batch test
    batch = np.stack([expr_high, expr_low])
    scores = ucell_score(batch, genes, sig)
    print(f"batch scores: {scores}")
    assert scores.shape == (2,)
    print("smoke tests passed")