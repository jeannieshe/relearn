"""
Utilities for mapping tahoebio/EmeraldBay's raw single-cell expression rows
(sparse gene-token-id / count lists, on EmeraldBay's own ~63k-token gene
vocabulary) onto STATE's fixed 2000-gene HVG panel, normalized the same way
STATE's `X_hvg` representation is normalized -- library-size-normalize each
cell to 1e4 total counts, then log1p. See `ucell_score`'s docstring in
`src/relearn/utils.py` for why this specific normalization matters: UCell
only uses within-cell rank order, so any monotonic transform of raw counts
gives the same score, but log1p vs. raw *does* change how ties are broken
at very low counts, so scores are only comparable to STATE's real X_hvg
predictions if real EmeraldBay cells go through the same transform.

Plain importable module -- no side effects on import, no `if __name__ ==
"__main__"` script logic. See `emeraldbay_overlap_check.py` for a sibling
script that does the drug-name-overlap analysis; the __main__ block at the
bottom of *this* file is validation-by-hand, run directly with:

    /home/jeannie/miniconda/envs/pytorch-pip/bin/python \\
        src/relearn/experiments/emeraldbay_utils.py

EmeraldBay's gene vocab (per its HF dataset card) extends Tahoe-100M's:
~62,710 token IDs preserved from Tahoe-100M's vocabulary, plus 574
EmeraldBay-specific genes appended after them -- `metadata/gene_metadata.parquet`
is the token_id -> gene_symbol/ensembl_id lookup table for the whole thing.

A design note on normalization: EmeraldBay's per-row `genes`/`expressions`
lists are each cell's *entire* measured set of nonzero genes (thousands of
entries, not capped to 2000) -- i.e. close to that cell's full transcriptome,
not a pre-subsetted HVG panel. STATE's own X_hvg pipeline normalizes over the
full transcriptome and *then* subsets to the 2000 HVG columns, so this module
mirrors that ordering: the per-cell library-size denominator is the sum of
ALL of that row's raw counts (every gene, whether or not it survives into the
HVG panel), not just the sum restricted to the ~2000 panel genes that make it
through `densify_rows`. Normalizing against the post-drop subset sum instead
would systematically inflate every value (a much smaller denominator for the
same numerator) and would not match what a real STATE X_hvg cell looks like.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).parent.parent.parent.parent
HVG_GENE_NAMES_PATH = "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy"
EMERALDBAY_REPO = "tahoebio/EmeraldBay"


def load_hvg_gene_names(path: str = HVG_GENE_NAMES_PATH) -> np.ndarray:
    """STATE's fixed 2000-gene HVG panel, as gene *symbols* (e.g. 'CFH',
    'GCLC', ...), in the column order the model's X_hvg obsm/output uses.
    Confirmed by inspection: `np.load(..., allow_pickle=True)` gives a
    length-2000 object ndarray of plain gene-symbol strings, not Ensembl
    IDs or token indices."""
    return np.load(path, allow_pickle=True)


def load_emeraldbay_gene_metadata(repo_id: str = EMERALDBAY_REPO) -> pd.DataFrame:
    """EmeraldBay's gene vocab table: token_id -> gene_symbol/ensembl_id.
    Columns confirmed by inspection: ['gene_symbol', 'ensembl_id',
    'token_id'], 63284 rows total (matches the ~62,710 Tahoe-100M-derived +
    574 EmeraldBay-specific gene count from the dataset card, ballpark)."""
    from huggingface_hub import hf_hub_download  # local import: only needed here

    path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename="metadata/gene_metadata.parquet"
    )
    return pq.read_table(path).to_pandas()


def build_token_to_hvg_index(
    gene_meta: pd.DataFrame, hvg_gene_names: np.ndarray
) -> tuple[dict[int, int], float]:
    """
    Build a mapping from EmeraldBay gene `token_id` -> column index into the
    2000-column HVG panel (in `hvg_gene_names` order). Matching is by exact
    gene_symbol string equality. Panel genes not found anywhere in
    EmeraldBay's vocab (and EmeraldBay-vocab genes not in the panel) are
    simply absent from the mapping -- they get dropped, not zero-filled with
    a wrong id.

    Returns
    -------
    token_to_col : dict[int, int]
        EmeraldBay token_id -> 0..1999 column index.
    coverage : float
        Fraction of the 2000 HVG panel genes that were actually found (by
        gene_symbol) in EmeraldBay's vocab. This is the number that matters
        for whether the STATE-vs-EmeraldBay comparison is meaningful at
        all -- low coverage means most of the reward signature's resolution
        is simply unobservable in EmeraldBay's expression data.
    """
    # A gene_symbol can repeat in gene_meta (e.g. remapped/withdrawn HGNC
    # symbols); keep the first occurrence deterministically.
    symbol_to_token: dict[str, int] = {}
    for sym, tok in zip(gene_meta["gene_symbol"], gene_meta["token_id"]):
        if sym not in symbol_to_token:
            symbol_to_token[sym] = int(tok)

    token_to_col: dict[int, int] = {}
    n_found = 0
    for col_idx, sym in enumerate(hvg_gene_names):
        tok = symbol_to_token.get(sym)
        if tok is not None:
            token_to_col[tok] = col_idx
            n_found += 1
    coverage = n_found / len(hvg_gene_names)
    return token_to_col, coverage


def densify_rows(
    genes_col: Iterable, expressions_col: Iterable, token_to_col: dict[int, int],
    n_genes: int = 2000,
) -> np.ndarray:
    """
    Densify a batch of EmeraldBay rows' sparse (genes, expressions) lists
    onto the 2000-column HVG panel. `genes_col`/`expressions_col` are
    per-row sequences straight off a parquet file's 'genes'/'expressions'
    columns (each element is itself a list/array of token ids or raw
    counts for one cell). Genes not in `token_to_col` are dropped. Returns
    RAW (un-normalized) densified counts, shape [n_cells, n_genes]; missing
    genes are 0.
    """
    genes_col = list(genes_col)
    expressions_col = list(expressions_col)
    n_cells = len(genes_col)
    dense = np.zeros((n_cells, n_genes), dtype=np.float64)
    for i, (genes, exprs) in enumerate(zip(genes_col, expressions_col)):
        genes = np.asarray(genes)
        exprs = np.asarray(exprs, dtype=np.float64)
        for tok, val in zip(genes, exprs):
            col = token_to_col.get(int(tok))
            if col is not None:
                dense[i, col] = val
    return dense


def library_sizes(expressions_col: Iterable) -> np.ndarray:
    """Per-cell total raw counts, summed over EVERY gene in that row (not
    just the ones that survive into the HVG panel) -- the correct
    denominator for library-size normalization; see module docstring."""
    return np.array([float(np.sum(e)) for e in expressions_col])


def normalize_like_state_hvg(
    dense_counts: np.ndarray, lib_sizes: np.ndarray, target_sum: float = 1e4,
) -> np.ndarray:
    """
    Normalize densified HVG-panel counts the way STATE's X_hvg is
    normalized: library-size-normalize each cell to `target_sum` total
    counts using that cell's TRUE (whole-transcriptome) library size, then
    log1p. `lib_sizes` must be computed from the full row (use
    `library_sizes()` on the *un-subsetted* expressions column), not
    re-derived from `dense_counts.sum(axis=1)` -- the latter would only sum
    the post-drop HVG subset and understate the true library size.
    """
    lib_sizes = np.asarray(lib_sizes, dtype=np.float64)
    safe_lib = np.where(lib_sizes == 0, 1.0, lib_sizes)
    scale = (target_sum / safe_lib)[:, None]
    return np.log1p(dense_counts * scale)


def densify_and_normalize(
    df: pd.DataFrame, token_to_col: dict[int, int], n_genes: int = 2000,
) -> np.ndarray:
    """
    Convenience wrapper: given a dataframe with 'genes'/'expressions'
    columns (as read directly from an EmeraldBay parquet file), return the
    library-size-normalized + log1p'd [n_cells, n_genes] matrix, ready to
    hand to `utils.ucell_score` alongside `load_hvg_gene_names()`.
    """
    dense = densify_rows(df["genes"], df["expressions"], token_to_col, n_genes)
    lib_sizes = library_sizes(df["expressions"])
    return normalize_like_state_hvg(dense, lib_sizes)


if __name__ == "__main__":
    # Hand validation against the 3 EmeraldBay parquet files pulled so far.
    # Not a formal test suite (there isn't one in this repo, per CLAUDE.md) --
    # just confirms the pipeline runs end-to-end on real data and reports
    # sanity numbers a reader can eyeball.
    hvg_names = load_hvg_gene_names()
    print(f"HVG panel: {len(hvg_names)} gene symbols, e.g. {list(hvg_names[:5])}")

    gene_meta = load_emeraldbay_gene_metadata()
    print(f"EmeraldBay gene_metadata: {len(gene_meta)} rows, columns "
          f"{gene_meta.columns.tolist()}")

    token_to_col, coverage = build_token_to_hvg_index(gene_meta, hvg_names)
    print(f"\nHVG panel coverage in EmeraldBay's vocab: {coverage:.1%} "
          f"({len(token_to_col)}/{len(hvg_names)} panel genes found)")

    data_dir = REPO_ROOT / "data/datasets/EmeraldBay/SW480"
    for name in ["dmso_control", "gemcitabine_paclitaxel", "dabrafenib_trametinib"]:
        path = data_dir / f"{name}.parquet"
        if not path.exists():
            print(f"\n{name}: file not found, skipping")
            continue
        df = pq.read_table(path).to_pandas()
        normalized = densify_and_normalize(df, token_to_col)
        n_nonzero_genes = (normalized > 0).sum(axis=1)
        print(f"\n{name}: {len(df)} cells -> normalized shape {normalized.shape}")
        print(f"  value range: min={normalized.min():.4f} max={normalized.max():.4f} "
              f"mean(nonzero)={normalized[normalized > 0].mean():.4f}")
        print(f"  nonzero HVG-panel genes per cell: min={n_nonzero_genes.min()} "
              f"max={n_nonzero_genes.max()} mean={n_nonzero_genes.mean():.1f} "
              f"(out of {normalized.shape[1]})")
        assert np.isfinite(normalized).all(), "non-finite values in normalized output"
        assert normalized.min() >= 0, "normalized values should be non-negative"

    print("\nAll smoke tests passed.")
