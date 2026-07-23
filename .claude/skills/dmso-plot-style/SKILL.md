---
name: dmso-plot-style
description: >-
  House plotting style for the DMSO self-consistency figures in
  notebooks/jeannie/dmso_sweep_comparison.py -- colors, fonts, reference
  constants, and how each plot function is built. Use when adding a new
  plot to that notebook, restyling an existing one, or porting the same
  look to a different notebook/script.
---

# DMSO sweep comparison: plotting conventions

Source of truth: `notebooks/jeannie/dmso_sweep_comparison.py` (a marimo
notebook). This doc mirrors that notebook's style as of the session that
wrote it -- if the notebook has since diverged, trust the notebook.

## Fonts

Set once, globally, near the top of the notebook (right after the imports
cell):

```python
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.formatter.use_mathtext"] = True
```

- Regular text (titles, axis labels, tick labels, legends) renders in a
  Helvetica-alike. **True Helvetica and Arial are not installed** on this
  machine -- `font.sans-serif` lists them anyway as aspirational/portable
  fallbacks, but the fonts that actually resolve here are `Nimbus Sans`
  (URW's open-source Helvetica clone) and `TeX Gyre Heros` (its
  descendant). If you skip straight to `Helvetica`/`Arial`/`DejaVu Sans`,
  matplotlib silently falls back to DejaVu Sans, which looks visibly
  different -- always list the URW/TeX-Gyre names first.
- Math text (anything inside `$...$`, e.g. `r"DMSO $\rightarrow$ DMSO"`)
  renders via matplotlib's built-in `mathtext` with the `cm` (Computer
  Modern) fontset -- LaTeX-style equation typography with **no system
  LaTeX/dvipng install required**. This works identically on any machine.
- Before assuming a font name resolves, check what's actually installed:
  ```python
  import matplotlib.font_manager as fm
  fm.findfont("Nimbus Sans", fallback_to_default=False)  # raises if missing
  ```

## Shared color palette and constants

Defined once as notebook globals, reused across every plot function:

```python
GRIDLINE, MUTED, SECONDARY, PRIMARY = "#e1e0d9", "#898781", "#52514e", "#0b0b0b"
DMSO_RING = "#e34948"

UNRELATED_FLOOR = 0.780   # mean pairwise cosine sim across 200 random single-drug
                          # outcomes (19,900 pairs, seed=0) -- std=0.175, range
                          # 0.037-0.989, median 0.841. A reference floor, not a
                          # per-dataset statistic -- don't recompute it per-df.
NEUTRAL_SCORE = 0.5584786324786325  # UCell(HALLMARK_APOPTOSIS) of untreated SW480
```

Role of each color:

| Name | Hex | Used for |
|---|---|---|
| `PRIMARY` | `#0b0b0b` | Titles |
| `SECONDARY` | `#52514e` | Annotation/reference-line label text |
| `MUTED` | `#898781` | Axis spines, tick labels, axis label text, dashed reference lines |
| `GRIDLINE` | `#e1e0d9` | Grid lines, bottom/left spine color |
| `DMSO_RING` | `#e34948` | Open-circle highlight marking the DMSO-\>DMSO point specifically |
| `tab:orange` / `tab:blue` | (matplotlib defaults) | Series identity: "DMSO first, drug second" vs. "drug first, DMSO second" |

## Common axes chrome (every plot function repeats this)

```python
fig, ax = plt.subplots(figsize=(10, 6.5))   # (7, 6.5) for the narrower scatter-by-sweep plot
fig.patch.set_facecolor("white")

# ... plot-specific content ...

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(GRIDLINE)
ax.tick_params(colors=MUTED, labelsize=8.5)   # 9 on the categorical scatter plot
ax.grid(True, color=GRIDLINE, linewidth=0.8)  # axis="y" only on the categorical plot
ax.set_axisbelow(True)

fig.tight_layout()
return fig
```

Text label boxes that sit on top of data points use a consistent
translucent white background so they stay legible over scattered points:

```python
label_bg = dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2)
```

The DMSO-\>DMSO point is always highlighted the same way: an unfilled
ring (`facecolors="none"`, `edgecolors=DMSO_RING`) drawn *on top of*
(higher `zorder`) the regular scatter, at the exact same coordinates as
that point's own marker -- see the jitter gotcha below.

## Per-plot construction notes

### `plot_cosine_vs_score(df, specific_title)` -- single-sweep scatter

x = `score_drug_alone` (continuous), y = `cosine_sim`, color = `l2_dist`
(sequential `"Blues"` colormap + colorbar). Two vertical/horizontal
reference lines make sense here because both axes are continuous:
`UNRELATED_FLOOR` (horizontal) and `NEUTRAL_SCORE` (vertical). The 12
worst self-consistency offenders (`nsmallest(12, "cosine_sim")`) get
direct-labeled on a staggered 4-row shelf below the data so labels don't
collide; drug names are cleaned with
`re.sub(r"\s*\([^)]*\)\s*$", "", name)` to strip trailing parenthetical
annotations.

### `plot_cosine_vs_score_overlay(df_first, df_second)` -- both sweeps, one axes

Same idea, but color now carries *series identity* (`tab:orange` vs.
`tab:blue`) instead of `l2_dist`, since one plot can't carry both
channels without cluttering it -- the per-sweep `l2_dist`-colored view is
what the plot above is for. Only the top 3 worst offenders get labeled,
pooled across both sweeps combined (`pd.concat(...).nsmallest(3, ...)`),
not per-series, because the point of the overlay is one shared
comparison.

### `plot_cosine_scatter(df_first, df_second)` -- categorical, discrete points

Built when the request was "same comparison, but drop the x-axis
(potency) entirely." Two fixed x positions (`0`, `1`), one per sweep;
each point gets a random jitter offset instead of a real x-value, so
overlapping `cosine_sim` values stay visible as discrete dots (this was
deliberately **not** a violin/KDE plot -- an early version used
`ax.violinplot`, which the user explicitly rejected in favor of literal
scattered points).

```python
rng = np.random.default_rng(0)
jitter = rng.uniform(-0.18, 0.18, size=len(df))
xs = pos + jitter
```

**Jitter gotcha (bit us once):** the DMSO ring must be drawn at the
*same jittered x* as the DMSO row's own scatter point, not at the bare
category position `pos`. Compute `jitter`/`xs` once per dataset, find
the DMSO row's index into that same array, and reuse it -- don't call
`rng.uniform(...)` a second time in a separate loop for the ring, since
that draws a *different* random offset and the ring ends up circling
empty space next to the point instead of around it:

```python
dmso_mask = (df["drug"] == "[('DMSO_TF', 0.0, 'uM')]").to_numpy()
dmso_i = np.flatnonzero(dmso_mask)[0]
ax.scatter([xs[dmso_i]], [ys[dmso_i]], s=90, facecolors="none",
           edgecolors=DMSO_RING, linewidths=1.6, zorder=4)
```

No `UNRELATED_FLOOR` or `NEUTRAL_SCORE` lines here -- both were tied to
the continuous x-axis (drug potency) that this plot deliberately drops,
so they stopped being meaningful once the x-axis became categorical.
Instead there's a "ceiling" line at `max(df_first["cosine_sim"].max(),
df_second["cosine_sim"].max())` (empirically ~0.983 in both sweeps) --
notably this ceiling is achieved by the DMSO-\>DMSO point itself in both
sweeps, i.e. DMSO-alone-vs-DMSO-alone is the most self-consistent single
outcome in the data. X-axis ticks are made explicit with
`ax.tick_params(axis="x", length=5, width=0.8, colors=MUTED)` since a
2-category axis otherwise reads as bare labels with no tick marks.

## Reused identifiers across the notebook

The drug identity string for DMSO is always matched as the literal
`"[('DMSO_TF', 0.0, 'uM')]"` (the column is a stringified list-of-tuples,
not a real Python object -- parse with `ast.literal_eval(r["drug"])[0]`
when you need the drug name/concentration/units out of it, as the
worst-offender labeling does).
