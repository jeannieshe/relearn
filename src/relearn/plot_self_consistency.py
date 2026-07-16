"""
Static PDF rendering of the self-consistency sweep (experiments/self_consistency_ranking.csv):
does f(output_A, DMSO) ~ output_A? See self_consistency_check.py for how that CSV is produced.

Page 1: scatter of single-drug effect (x) vs. cosine similarity between drug-A-alone
and drug-A+DMSO states (y), colored by L2 distance between those same two states.
Page 2: the worst/best self-consistency tables.
"""

import ast
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).parent.parent.parent
IN_PATH = REPO_ROOT / "experiments" / "self_consistency_ranking.csv"
OUT_PATH = REPO_ROOT / "experiments" / "self_consistency_ranking.pdf"

UNRELATED_FLOOR = 0.83  # cosine sim between two unrelated single-drug outcomes
NEUTRAL_SCORE = 0.5584786324786325  # UCell score of the untreated SW480 baseline

# same sequential blue ramp used in the interactive artifact version of this chart
SEQ_LOW, SEQ_HIGH = "#b7d3f6", "#0d366b"
GRIDLINE, MUTED, SECONDARY, PRIMARY = "#e1e0d9", "#898781", "#52514e", "#0b0b0b"
DMSO_RING = "#e34948"


def _short_label(drug_str: str) -> str:
    name, conc, units = ast.literal_eval(drug_str)[0]
    return f"{name} ({conc:g}{units})"


def load_rows(path: Path) -> list:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["score_A"] = float(r["score_A"])
        r["cosine_sim"] = float(r["cosine_sim"])
        r["l2_dist"] = float(r["l2_dist"])
    return rows


def plot_scatter(ax, rows):
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SEQ_LOW, SEQ_HIGH])
    xs = [r["score_A"] for r in rows]
    ys = [r["cosine_sim"] for r in rows]
    l2s = [r["l2_dist"] for r in rows]

    sc = ax.scatter(xs, ys, c=l2s, cmap=cmap, s=26, linewidths=0.6, edgecolors="white", zorder=3)

    dmso = next(r for r in rows if r["drug"] == "[('DMSO_TF', 0.0, 'uM')]")
    ax.scatter([dmso["score_A"]], [dmso["cosine_sim"]], s=90, facecolors="none",
               edgecolors=DMSO_RING, linewidths=1.6, zorder=4)
    ax.annotate("DMSO → DMSO", (dmso["score_A"], dmso["cosine_sim"]),
                xytext=(8, 6), textcoords="offset points", fontsize=9,
                color=DMSO_RING, fontweight="bold")

    label_bg = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2)

    ax.axhline(UNRELATED_FLOOR, color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(max(xs), UNRELATED_FLOOR + 0.012,
             "cosine of two unrelated single-drug outcomes (≈ 0.83)",
             ha="right", va="bottom", fontsize=8.5, color=SECONDARY, zorder=5, bbox=label_bg)

    ax.axvline(NEUTRAL_SCORE, color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(NEUTRAL_SCORE + 0.0015, max(ys) - 0.01,
             f"untreated SW480, no drug ({NEUTRAL_SCORE:.3f})",
             ha="left", va="top", fontsize=8.5, color=SECONDARY, zorder=5, bbox=label_bg)

    # Direct-label the worst offenders -- the strongest cytotoxics, whose
    # self-consistency breaks down most (see EXPERIMENTS.md). These 12 points
    # sit in a narrow x-band, too tight for inline labels without collisions,
    # so they're pinned to a 4-row shelf below the data with straight leader
    # lines. Row assignment alternates by x-order (idx % 4) so neighbors in x
    # never land in the same row.
    worst = sorted(sorted(rows, key=lambda r: r["cosine_sim"])[:12], key=lambda r: r["score_A"])
    y_bottom = min(ys)  # lowest real data point (~0.137) -- shelf lives below this, above 0
    n_rows = 4
    row_ys = [y_bottom * (0.82 - 0.68 * i / (n_rows - 1)) for i in range(n_rows)]
    ax.set_ylim(-0.03, ax.get_ylim()[1])

    for i, r in enumerate(worst):
        name, conc, _units = ast.literal_eval(r["drug"])[0]
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name)  # drop salt/formulation suffix, e.g. "(hydrochloride)"
        label = f"{name} ({conc:g}µM)" if name == "Homoharringtonine" else name
        row_y = row_ys[i % n_rows]
        ax.plot([r["score_A"], r["score_A"]], [r["cosine_sim"] - 0.012, row_y + 0.012],
                color=MUTED, linewidth=0.7, zorder=4)
        ax.text(r["score_A"], row_y, label, ha="center", va="center", fontsize=7.2,
                color=SECONDARY, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))

    ax.set_xlabel("Single-drug effect  —  UCell apoptosis score of drug A alone", fontsize=10, color=MUTED)
    ax.set_ylabel("Cosine similarity  —  drug A alone  vs.  drug A + DMSO", fontsize=10, color=MUTED)
    ax.set_title("Self-consistency vs. drug potency", fontsize=13, fontweight="600", color=PRIMARY, loc="left")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)

    cbar = ax.figure.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("L2 distance", fontsize=9, color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_visible(False)


def plot_tables(fig, rows):
    sorted_rows = sorted(rows, key=lambda r: r["cosine_sim"])
    worst, best = sorted_rows[:12], sorted_rows[-12:][::-1]

    def as_table_data(subset):
        return [[r["drug"].strip("[]").replace("'", ""), f"{r['score_A']:.4f}",
                 f"{r['cosine_sim']:.4f}", f"{r['l2_dist']:.4f}"] for r in subset]

    for ax, title, subset in (
        (fig.add_subplot(1, 2, 1), "Worst self-consistency (lowest cosine)", worst),
        (fig.add_subplot(1, 2, 2), "Best self-consistency (highest cosine)", best),
    ):
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="600", color=PRIMARY, loc="left", pad=14)
        table = ax.table(
            cellText=as_table_data(subset),
            colLabels=["Drug", "Score A", "Cosine", "L2"],
            bbox=[0, 0.35, 1, 0.62], cellLoc="left", colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.auto_set_column_width(col=list(range(4)))
        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor(GRIDLINE)
            if row == 0:
                cell.set_text_props(fontweight="600", color=MUTED)
                cell.set_facecolor("white")


if __name__ == "__main__":
    rows = load_rows(IN_PATH)

    with PdfPages(OUT_PATH) as pdf:
        fig1, ax1 = plt.subplots(figsize=(10, 6.5))
        fig1.patch.set_facecolor("white")
        plot_scatter(ax1, rows)
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = plt.figure(figsize=(11, 6.5))
        fig2.patch.set_facecolor("white")
        plot_tables(fig2, rows)
        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

    print(f"Wrote {OUT_PATH}")
