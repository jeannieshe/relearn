# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.14",
#     "matplotlib==3.11.1",
#     "numpy==2.5.1",
#     "pandas==3.0.5",
#     "scipy==1.18.0",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return Path, ast, mo, np, pd, plt, re


@app.cell
def _(plt):
    # Helvetica for regular text; Computer Modern is kept for math via
    # mathtext.fontset so "$...$" equations still render LaTeX-style.
    # True Helvetica isn't installed on this machine, so we point at the
    # open-source URW/TeX-Gyre clones (metrically near-identical) instead
    # of silently falling back to DejaVu Sans.
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.formatter.use_mathtext"] = True
    return


@app.cell
def _(mo):
    mo.md(r"""
    # DMSO sweep comparison

    Interactive analysis comparing the two DMSO-position sweeps:

    - `dmso_first_ranking.csv` — DMSO fixed **first**, drug swept **second**
    - `dmso_second_ranking.csv` — drug swept **first**, DMSO fixed **second**
      (also the self-consistency sweep: does `f(drug, DMSO) ~ f(drug)` alone?)

    Starting with the cosine-similarity-vs-drug-potency graph, ported from
    `src/relearn/scripts/plot_self_consistency.py`.
    """)
    return


@app.cell
def _(Path):
    REPO_ROOT = Path(__file__).parent.parent.parent
    DMSO_SECOND_PATH = REPO_ROOT / "experiments" / "dmso_second_ranking.csv"
    DMSO_FIRST_PATH = REPO_ROOT / "experiments" / "dmso_first_ranking.csv"
    return DMSO_FIRST_PATH, DMSO_SECOND_PATH


@app.cell
def _(DMSO_FIRST_PATH, DMSO_SECOND_PATH, pd):
    dmso_second_df = pd.read_csv(DMSO_SECOND_PATH)
    dmso_first_df = pd.read_csv(DMSO_FIRST_PATH)
    dmso_second_df.head()
    return dmso_first_df, dmso_second_df


@app.cell
def _(dmso_first_df):
    dmso_dmso= dmso_first_df[dmso_first_df['drug'].str.contains('DMSO', na=False)]
    return (dmso_dmso,)


@app.cell
def _(dmso_dmso):
    dmso_dmso
    return


@app.cell
def _(dmso_first_df):
    dmso_first_df['cosine_sim'].max()
    return


@app.cell
def _(dmso_first_df):
    # cosine_sim / l2_dist now come straight from the CSV -- dmso_first_sweep.py
    # was updated to compute them against the fixed post-DMSO reference state
    # (the mirror of dmso_second's drug-alone reference), so there was no way
    # to derive them from the score-only columns alone; had to re-run the sweep.
    dmso_first_df.head()
    return


@app.cell
def _():
    # mean pairwise cosine similarity across 200 random single-drug outcomes
    # (19,900 pairs; seed=0) -- NOT a single anecdotal pair. Wide spread:
    # std=0.175, range 0.037-0.989, median 0.841 (a single pair is noisy).
    UNRELATED_FLOOR = 0.780
    NEUTRAL_SCORE = 0.5584786324786325  # UCell score of the untreated SW480 baseline

    GRIDLINE, MUTED, SECONDARY, PRIMARY = "#e1e0d9", "#898781", "#52514e", "#0b0b0b"
    DMSO_RING = "#e34948"
    return (
        DMSO_RING,
        GRIDLINE,
        MUTED,
        NEUTRAL_SCORE,
        PRIMARY,
        SECONDARY,
        UNRELATED_FLOOR,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Self-consistency: cosine similarity vs. single-drug effect
    """)
    return


@app.cell
def _(
    DMSO_RING,
    GRIDLINE,
    MUTED,
    NEUTRAL_SCORE,
    PRIMARY,
    SECONDARY,
    UNRELATED_FLOOR,
    ast,
    plt,
    re,
):
    def plot_cosine_vs_score(df, specific_title):
        """Cosine similarity (drug alone vs. drug+DMSO) against single-drug
        potency, colored by L2 distance -- same design as plot_self_consistency.py."""
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor("white")

        xs = df["score_drug_alone"].to_numpy()
        ys = df["cosine_sim"].to_numpy()
        l2s = df["l2_dist"].to_numpy()

        sc = ax.scatter(xs, ys, c=l2s, cmap="Blues", s=26, linewidths=0.6, edgecolors="white", zorder=3)

        dmso_row = df.loc[df["drug"] == "[('DMSO_TF', 0.0, 'uM')]"].iloc[0]
        ax.scatter([dmso_row["score_drug_alone"]], [dmso_row["cosine_sim"]], s=90, facecolors="none",
                   edgecolors=DMSO_RING, linewidths=1.6, zorder=4)
        ax.annotate(r"DMSO $\rightarrow$ DMSO", (dmso_row["score_drug_alone"], dmso_row["cosine_sim"]),
                    xytext=(8, 6), textcoords="offset points", fontsize=9,
                    color=DMSO_RING, fontweight="bold")

        label_bg = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2)

        ax.axhline(UNRELATED_FLOOR, color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
        ax.text(xs.max(), UNRELATED_FLOOR + 0.012,
                r"mean cosine of two unrelated single-drug outcomes ($\approx$ 0.78)",
                ha="right", va="bottom", fontsize=8.5, color=SECONDARY, zorder=5, bbox=label_bg)

        ax.axvline(NEUTRAL_SCORE, color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
        ax.text(NEUTRAL_SCORE + 0.0015, ys.max() - 0.01,
                f"untreated SW480, no drug ({NEUTRAL_SCORE:.3f})",
                ha="left", va="top", fontsize=8.5, color=SECONDARY, zorder=5, bbox=label_bg)

        # direct-label the worst self-consistency offenders on a staggered
        # shelf below the data -- see plot_self_consistency.py for why
        worst = df.nsmallest(12, "cosine_sim").sort_values("score_drug_alone")
        y_bottom = ys.min()
        n_rows = 4
        row_ys = [y_bottom * (0.82 - 0.68 * i / (n_rows - 1)) for i in range(n_rows)]
        ax.set_ylim(-0.03, ax.get_ylim()[1])

        for i, (_, r) in enumerate(worst.iterrows()):
            name, conc, _units = ast.literal_eval(r["drug"])[0]
            name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
            label = rf"{name} ({conc:g}$\mu$M)" if name == "Homoharringtonine" else name
            row_y = row_ys[i % n_rows]
            ax.plot([r["score_drug_alone"], r["score_drug_alone"]], [r["cosine_sim"] - 0.012, row_y + 0.012],
                    color=MUTED, linewidth=0.7, zorder=4)
            ax.text(r["score_drug_alone"], row_y, label, ha="center", va="center", fontsize=7.2,
                    color=SECONDARY, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))

        ax.set_xlabel("Single-drug effect: UCell(HALLMARK_APOPTOSIS)", fontsize=10, color=MUTED)
        ax.set_ylabel("Cosine similarity: ST(cell + drug) vs. ST(ST(cell + drug) + DMSO)", fontsize=10, color=MUTED)
        ax.set_title(f"Self-consistency vs. drug potency: {specific_title}", fontsize=13, fontweight="600", color=PRIMARY) #, loc="left")

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRIDLINE)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.grid(True, color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)

        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("L2 distance", fontsize=9, color=MUTED)
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
        cbar.outline.set_visible(False)

        fig.tight_layout()
        return fig

    return (plot_cosine_vs_score,)


@app.cell
def _(dmso_second_df, plot_cosine_vs_score):
    plot_cosine_vs_score(dmso_second_df, specific_title="drug first, DMSO second")
    return


@app.cell
def _(dmso_first_df, plot_cosine_vs_score):
    plot_cosine_vs_score(dmso_first_df, "DMSO first, drug second")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Overlay: DMSO first vs. DMSO second
    """)
    return


@app.cell
def _(
    GRIDLINE,
    MUTED,
    NEUTRAL_SCORE,
    PRIMARY,
    SECONDARY,
    UNRELATED_FLOOR,
    ast,
    pd,
    plt,
    re,
):
    def plot_cosine_vs_score_overlay(df_first, df_second):
        """Both sweeps on one axes for direct comparison. Color now carries
        series identity (which sweep), not L2 distance -- can't do both at
        once without cluttering the channel, so the per-dataset L2-colored
        plots above are the place for that view."""
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor("white")

        ax.scatter(df_first["score_drug_alone"], df_first["cosine_sim"],
                   s=26, alpha=0.55, color="tab:orange", edgecolors="white",
                   linewidths=0.4, zorder=3, label="DMSO first, drug second")
        ax.scatter(df_second["score_drug_alone"], df_second["cosine_sim"],
                   s=26, alpha=0.55, color="tab:blue", edgecolors="white",
                   linewidths=0.4, zorder=3, label="drug first, DMSO second")

        label_bg = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2)
        xmax = max(df_first["score_drug_alone"].max(), df_second["score_drug_alone"].max())

        ax.axhline(UNRELATED_FLOOR, color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
        ax.text(xmax, UNRELATED_FLOOR + 0.012,
                r"mean cosine of two unrelated single-drug outcomes ($\approx$ 0.78)",
                ha="right", va="bottom", fontsize=8.5, color=SECONDARY, zorder=5, bbox=label_bg)

        ax.axvline(NEUTRAL_SCORE, color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
        ax.text(NEUTRAL_SCORE + 0.0015, 0.02,
                f"untreated SW480, no drug ({NEUTRAL_SCORE:.3f})",
                ha="left", va="bottom", fontsize=8.5, color=SECONDARY, zorder=5, bbox=label_bg)

        # label the top 3 worst self-consistency offenders, pooled across
        # both sweeps combined (not per-series) -- the point of the overlay
        # is a direct, single comparison, not two separate worst-lists
        combined = pd.concat([df_first, df_second], ignore_index=True)
        worst3 = combined.nsmallest(3, "cosine_sim").sort_values("score_drug_alone")
        label_y = combined["cosine_sim"].min() - 0.09
        ax.set_ylim(label_y - 0.05, 1.03)

        for _, r in worst3.iterrows():
            name, conc, _units = ast.literal_eval(r["drug"])[0]
            name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
            label = rf"{name} ({conc:g}$\mu$M)"
            ax.plot([r["score_drug_alone"], r["score_drug_alone"]], [r["cosine_sim"] - 0.012, label_y + 0.012],
                    color=MUTED, linewidth=0.7, zorder=4)
            ax.text(r["score_drug_alone"], label_y, label, ha="center", va="center", fontsize=7.5,
                    color=SECONDARY, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))

        ax.set_xlabel("Single-drug effect  --  UCell apoptosis score of drug alone", fontsize=10, color=MUTED)
        ax.set_ylabel("Cosine similarity  --  drug alone  vs.  drug + DMSO", fontsize=10, color=MUTED)
        ax.set_title("Self-consistency vs. drug potency  --  DMSO first vs. second", fontsize=13, fontweight="600", color=PRIMARY, loc="left")

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRIDLINE)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.grid(True, color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)

        legend = ax.legend(loc="lower right", frameon=True, fontsize=9)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor(GRIDLINE)

        fig.tight_layout()
        return fig

    return (plot_cosine_vs_score_overlay,)


@app.cell
def _(dmso_first_df, dmso_second_df, plot_cosine_vs_score_overlay):
    plot_cosine_vs_score_overlay(dmso_first_df, dmso_second_df)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Self-consistency distribution: scatter by sweep

    Same comparison as the plots above, but with drug potency (the x-axis)
    dropped entirely -- discrete jittered points per sweep (not a continuous
    violin/density shape) showing the spread of `cosine_sim`.
    """)
    return


@app.cell
def _(DMSO_RING, GRIDLINE, MUTED, PRIMARY, SECONDARY, np, plt):
    def plot_cosine_scatter(df_first, df_second):
        """Same self-consistency comparison as the scatter/overlay plots above,
        but with drug potency (the x-axis) dropped -- discrete jittered points
        per sweep, not a continuous violin/density shape."""
        fig, ax = plt.subplots(figsize=(7, 6.5))
        fig.patch.set_facecolor("white")

        rng = np.random.default_rng(0)
        positions = [0, 1]
        datasets = [df_first, df_second]
        colors = ["tab:orange", "tab:blue"]
        label_bg = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2)

        for pos, df, color in zip(positions, datasets, colors):
            jitter = rng.uniform(-0.18, 0.18, size=len(df))
            xs = pos + jitter
            ys = df["cosine_sim"].to_numpy()
            ax.scatter(xs, ys, s=18, alpha=0.5, color=color, edgecolors="white", linewidths=0.3, zorder=3)

            dmso_mask = (df["drug"] == "[('DMSO_TF', 0.0, 'uM')]").to_numpy()
            dmso_i = np.flatnonzero(dmso_mask)[0]
            ax.scatter([xs[dmso_i]], [ys[dmso_i]], s=90, facecolors="none",
                       edgecolors=DMSO_RING, linewidths=1.6, zorder=4)

        ceiling = max(df_first["cosine_sim"].max(), df_second["cosine_sim"].max())
        ax.axhline(ceiling, color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
        ax.text(1.6, ceiling + 0.04, f"DMSO + DMSO cosine similarity ceiling at {ceiling:.3f}",
                ha="right", va="top", fontsize=8.5, color=SECONDARY, zorder=5, bbox=label_bg)

        ax.set_xlim(-0.7, 1.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(["DMSO first,\ndrug second", "drug first,\nDMSO second"])
        ax.tick_params(axis="x", length=5, width=0.8, colors=MUTED)
        ax.set_ylabel("Cosine similarity between f(drug) and f(drug with DMSO)", fontsize=10, color=MUTED)
        ax.set_title("Self-consistency of STATE under 2-step drug sequence: DMSO first vs. second", fontsize=13, fontweight="700", color=PRIMARY)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRIDLINE)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)

        fig.tight_layout()
        return fig


    return (plot_cosine_scatter,)


@app.cell
def _(dmso_first_df, dmso_second_df, plot_cosine_scatter):
    plot_cosine_scatter(dmso_first_df, dmso_second_df)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Similarity between sweeps: correlation across drugs

    Merge `dmso_first_df` and `dmso_second_df` on `drug`, then correlate each
    metric (`cosine_sim`, `score_drug_alone`) between the two sweeps across all
    shared drugs -- do the two sweep orders agree on *which* drugs are more or
    less self-consistent?
    """)
    return


@app.cell
def _(dmso_first_df, dmso_second_df):
    sweep_merged_df = dmso_first_df.merge(
        dmso_second_df, on="drug", suffixes=("_first", "_second")
    )

    cosine_sim_pearson = sweep_merged_df["cosine_sim_first"].corr(
        sweep_merged_df["cosine_sim_second"], method="pearson"
    )
    cosine_sim_spearman = sweep_merged_df["cosine_sim_first"].corr(
        sweep_merged_df["cosine_sim_second"], method="spearman"
    )
    score_drug_alone_pearson = sweep_merged_df["score_drug_alone_first"].corr(
        sweep_merged_df["score_drug_alone_second"], method="pearson"
    )
    score_drug_alone_spearman = sweep_merged_df["score_drug_alone_first"].corr(
        sweep_merged_df["score_drug_alone_second"], method="spearman"
    )

    print(
        f"cosine_sim (n={len(sweep_merged_df)} shared drugs): "
        f"Pearson r = {cosine_sim_pearson:.4f}, Spearman rho = {cosine_sim_spearman:.4f}"
    )
    print(
        f"score_drug_alone: "
        f"Pearson r = {score_drug_alone_pearson:.4f}, Spearman rho = {score_drug_alone_spearman:.4f}"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
