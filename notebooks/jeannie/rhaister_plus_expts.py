import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    import glob
    import os
    import warnings

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    return ast, glob, mo, np, os, pd, plt, spearmanr, warnings


@app.cell
def _(mo):
    mo.md(r"""
    # Rhaister-plus: combination-prediction experiments on EmeraldBay

    Marimo version of `rhaister_plus_expts.py` — the clean, self-contained,
    **reproducible** re-run of the exploratory analysis first developed in
    `view_tahoe_data.py`. Everything is regenerated *from source* (the
    EmeraldBay table + the raw single-cell h5ads + a curated design-class
    CSV); it does **not** read the cached `bars_*` / `ab_ceiling_result`
    files. Run with `marimo edit rhaister_plus_expts.py` in the
    `pytorch-pip` conda env (where `rhaister` + `h5py` are installed).

    ## Background
    The Rhaister *sensitivity* model predicts a cell line's `growth_rate`
    response to a drug, `growth_rate = log2(P_treated / P_DMSO)` (0 = no
    effect, negative = suppressed/killed). As shipped, the fewshot model is
    (1) an **ALS** additive decomposition `mu + cell_effect + treatment_effect`,
    (2) a **per-drug ridge** that shrinks combination weights toward 0, and
    (3) an ALS fallback. A combination A+B enters the ridge as one prior
    column `w_A·e_A + w_B·e_B`; the prior weights `W_prior` and shrinkage
    `lam` are the knobs these experiments probe. Two anchors matter:
    *shrink toward 0* ("Rhaister ships" → ignore ingredients) vs *shrink
    toward [1,1]* ("additivity" → combo = sum of its singles).

    ## Data / prerequisites
    1. **EmeraldBay** — `load_data("EmeraldBay")` with `HP_KEEP_MULTI_DRUG=1`
       (52 cell lines × 92 conditions = 4784 rows; auto-downloads from HF).
    2. **Cell-line metadata** — names / Organ / driver genes, from the HF snapshot.
    3. **Raw h5ads** (`H5_DIR`) — only per-cell `obs` codes are read (not the
       28 GB matrices) to recompute the A/B noise ceiling. Skipped with a
       warning if the path is absent; everything else still runs.
    4. **`artifacts/emeraldbay_combo_design_classes.csv`** — each combo →
       mechanistic `design_class`, used to colour the per-perturbation plots.

    ## Eligible test set
    A combo is estimator-eligible only if **both** components exist as exact
    single-agent (drug, dose) conditions (so an additive prior can be built):
    17 of 47 qualify, all 2-drug. Every panel experiment predicts these 17
    combos, holding out one cell line at a time (52 × 17 = 884 points).

    ## The four "bars"  (prior × intercept, `lam=10` unless noted)
    | bar | description |
    |---|---|
    | **bar1** | add-the-singles — `y(hc,A)+y(hc,B)` (pure additivity, no fit) |
    | **bar2** | "Rhaister ships" — ridge, shrink → 0, **with** free intercept |
    | **bar3** | [1,1] prior — ridge, shrink → [1,1] on ingredients, **no** intercept |
    | **bar4** | [1,1] + interaction — bar3's prior **with** a free intercept |

    Two ridge implementations are reproduced because the original consumed
    both: `fit_bars_2x2` (explicit prior×intercept ridge; drives Fig 1a and
    the bar3-vs-bar4 scatter) and `fit_bars_all52` (Rhaister's own
    `_drug_regression`; drives the potency / design-class / W_prior /
    baseline experiments). They agree closely (both: 20/52 cells bar3 R²<0).
    """)
    return


@app.cell
def _(os):
    # ---- configuration ----
    HERE = "/home/jeannie/relearn/notebooks/jeannie"
    ARTIFACTS = "/home/jeannie/relearn/artifacts"
    DESIGN_CLASS_CSV = os.path.join(ARTIFACTS, "emeraldbay_combo_design_classes.csv")
    H5_DIR = "/large_storage/goodarzilab/bioreason_cell/emeraldbay/h5ads"

    SW480 = "CVCL_0546"   # reference cell line
    LAM = 10.0            # ridge strength for the panel experiments
    AB_SEEDS = 5          # random half-split seeds for the noise ceiling

    # one colour per combo design class (used by every per-perturbation plot)
    CLASS_COLORS = {
        "horizontal bypass": "#4C72B0",
        "vertical blockade": "#DD8452",
        "same process / different mechanism": "#55A868",
        "orthogonal process": "#C44E52",
        "pharmacokinetic modulation": "#8172B3",
        "same-node redundancy": "#937860",
    }
    return AB_SEEDS, CLASS_COLORS, DESIGN_CLASS_CSV, H5_DIR, HERE, LAM, SW480


@app.cell
def _(DESIGN_CLASS_CSV, HERE, ast, glob, os, pd):
    # ======================================================================
    # Data loaders + shared index over the eligible combos
    # ======================================================================
    def fig_path(name):
        return os.path.join(HERE, name)

    def load_emeraldbay():
        """EmeraldBay sensitivity table (singles AND combos). One row per
        (cell_line, condition). HP_KEEP_MULTI_DRUG=1 must be set before
        importing load_data or combination rows are dropped."""
        os.environ["HP_KEEP_MULTI_DRUG"] = "1"
        from rhaister.prepare_sensitivity import load_data
        return load_data("EmeraldBay")

    def load_metadata():
        """Cell-line metadata + summary stats from the HF EmeraldBay snapshot."""
        pats = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--tahoebio--EmeraldBay/snapshots/*"))
        if not pats:
            raise FileNotFoundError("EmeraldBay HF snapshot not found; load_emeraldbay() first.")
        snap = pats[0]
        cell_line_md = pd.read_parquet(os.path.join(snap, "metadata/cell_line_metadata.parquet"))
        summary_stats = pd.read_parquet(os.path.join(snap, "metadata/summary_statistics.parquet"))
        return cell_line_md, summary_stats

    def load_design_classes():
        """Curated combo -> design_class mapping (plus drugs, doses_uM, moa)."""
        return pd.read_csv(DESIGN_CLASS_CSV)

    def name_lookup(cell_line_md):
        """CVCL id -> (cell_name, Organ), one row per cell line."""
        return (cell_line_md.drop_duplicates("Cell_ID_Cellosaur")
                .set_index("Cell_ID_Cellosaur")[["cell_name", "Organ"]])

    def build_index(eb):
        """Everything the leave-one-cell-out fits need: index maps, the 17
        eligible combos, their ingredient single-agent columns, and a
        growth_rate lookup."""
        from rhaister.combos import is_multi_drug
        cells = sorted(eb.cell_line.unique())
        treats = sorted(eb.condition.unique())
        c2i = {c: i for i, c in enumerate(cells)}
        t2i = {t: i for i, t in enumerate(treats)}
        single_cond = {}
        for c in treats:
            if not is_multi_drug(c):
                d, dose, _u = ast.literal_eval(c)[0]
                if not d.startswith("DMSO"):
                    single_cond[(d, dose)] = c
        eligible = [c for c in treats if is_multi_drug(c)
                    and all((d, dose) in single_cond for d, dose, _u in ast.literal_eval(c))]
        comp_cols = {c: [t2i[single_cond[(d, dose)]] for d, dose, _u in ast.literal_eval(c)]
                     for c in eligible}
        gr = {(r.cell_line, r.condition): r.growth_rate for r in eb.itertuples()}
        return dict(cells=cells, treats=treats, c2i=c2i, t2i=t2i, n_cell=len(cells),
                    n_treat=len(treats), eligible=eligible, comp_cols=comp_cols,
                    single_cond=single_cond, gr=gr, is_multi_drug=is_multi_drug)

    def r2_score(measured, pred):
        """Coefficient of determination (1 - SS_res/SS_tot); nan if no variance."""
        ss = ((measured - measured.mean()) ** 2).sum()
        return float("nan") if ss == 0 else float(1 - ((measured - pred) ** 2).sum() / ss)

    def per_perturbationr2_score(pred, pred_col):
        """Per combo: potency = mean measured growth_rate across the 52 cells,
        and R^2 of `pred_col` vs measured across those cells."""
        rows = []
        for cond, g in pred.groupby("condition"):
            m = g["measured"].to_numpy()
            row = {"condition": cond, "potency": float(m.mean()),
                   "r2": r2_score(m, g[pred_col].to_numpy())}
            if "drugs" in g.columns:
                row["drugs"] = g["drugs"].iloc[0]
            rows.append(row)
        return pd.DataFrame(rows)

    def attach_design_class(df, dc):
        """Add design_class / doses_uM keyed on the exact condition string."""
        return df.assign(
            design_class=df["condition"].map(dict(zip(dc.condition, dc.design_class))).fillna("unknown"),
            doses_uM=df["condition"].map(dict(zip(dc.condition, dc.doses_uM))).fillna(""),
        )

    return (
        attach_design_class,
        build_index,
        fig_path,
        load_design_classes,
        load_emeraldbay,
        load_metadata,
        name_lookup,
        per_perturbationr2_score,
        r2_score,
    )


@app.cell
def _(
    AB_SEEDS,
    H5_DIR,
    LAM,
    SW480,
    ast,
    build_index,
    glob,
    np,
    os,
    pd,
    r2_score,
    warnings,
):
    # ======================================================================
    # The fits: 2x2 ridge, Rhaister _drug_regression, W_prior sweep, A/B ceiling
    # ======================================================================
    def _ridge_2x2(Xn, xh, y, prior, use_intercept, lam):
        """One held-out prediction from an uncentered ridge over the held-out
        cell's own drug basis. `prior` is the shrink target per column;
        `use_intercept` appends an unpenalised constant column."""
        if use_intercept:
            Xa = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
            xa = np.append(xh, 1.0)
            wp = np.append(prior, 0.0)
            D = np.ones(Xa.shape[1]); D[-1] = 0.0
        else:
            Xa, xa, wp, D = Xn, xh, prior, np.ones(Xn.shape[1])
        W = np.linalg.solve(Xa.T @ Xa + lam * np.diag(D), Xa.T @ y + lam * (D * wp))
        return float(xa @ W)

    def fit_bars_2x2(eb, lam=LAM):
        """Leave-one-cell-out four bars via the explicit prior x intercept ridge.
        Returns {lam, pooled, per_cell_R2, cells, sw480_idx, pred}."""
        from rhaister.prepare_sensitivity import make_splits
        from rhaister.train_sensitivity import _als_decompose_scalar as als
        ix = build_index(eb)
        cells, t2i, c2i = ix["cells"], ix["t2i"], ix["c2i"]
        eligible, comp_cols, gr = ix["eligible"], ix["comp_cols"], ix["gr"]
        n_cell, n_treat = ix["n_cell"], ix["n_treat"]
        rows = []
        for hcn in cells:
            hc = c2i[hcn]
            tr, _ = make_splits(eb, {"holdout_cells": [hcn], "test_treatments": {hcn: set(eligible)}})
            ctr = np.array([c2i[c] for c in tr.cell_line]); ttr = np.array([t2i[t] for t in tr.condition])
            yv = tr.growth_rate.to_numpy(float)
            mu, ce, te = als(yv, ctr, ttr, n_cell, n_treat, n_iter=30)
            yimp = mu + ce[:, None] + te[None, :]; yimp[ctr, ttr] = yv
            nonh = np.array([i for i in range(n_cell) if i != hc])
            Dx = sorted({t2i[t] for t in tr.loc[tr.cell_line == hcn, "condition"]})
            pos = {t: j for j, t in enumerate(Dx)}
            X = yimp[np.ix_(nonh, Dx)]; xh = yimp[hc, Dx]
            for c in eligible:
                cols = [pos[g] for g in comp_cols[c]]
                y = yimp[nonh, t2i[c]]
                wp = np.zeros(len(Dx))
                for ci in cols:
                    wp[ci] = 1.0
                rows.append({
                    "cell_line": hcn, "condition": c,
                    "drugs": "+".join(d for d, _, _ in ast.literal_eval(c)),
                    "measured": gr[(hcn, c)],
                    "bar1": float(yimp[hc, comp_cols[c]].sum()),
                    "bar2": _ridge_2x2(X, xh, y, np.zeros(len(Dx)), True, lam),
                    "bar3": _ridge_2x2(X, xh, y, wp, False, lam),
                    "bar4": _ridge_2x2(X, xh, y, wp, True, lam),
                })
        pred = pd.DataFrame(rows)
        m = pred.measured.to_numpy()
        bars = ["bar1", "bar2", "bar3", "bar4"]
        pooled = {b: {"R2": round(r2_score(m, pred[b].to_numpy()), 4),
                      "Pearson": round(float(np.corrcoef(m, pred[b].to_numpy())[0, 1]), 4)}
                  for b in bars}
        per_cell = {b: [] for b in bars}
        for _, g in pred.groupby("cell_line"):
            mm = g.measured.to_numpy()
            for b in bars:
                per_cell[b].append(round(r2_score(mm, g[b].to_numpy()), 4))
        return {"lam": lam, "pooled": pooled, "per_cell_R2": per_cell,
                "cells": cells, "sw480_idx": cells.index(SW480), "pred": pred}

    def fit_bars_all52(eb, lams=(1.0, 10.0)):
        """Leave-one-cell-out four bars using Rhaister's `_drug_regression`.
        One row per (cell, combo): bar1_add, bar4_interaction, bar{2,3}_*_lam{lam}."""
        from rhaister.prepare_sensitivity import make_splits
        from rhaister.train_sensitivity import _als_decompose_scalar as als, _drug_regression as dreg
        ix = build_index(eb)
        cells, t2i, c2i = ix["cells"], ix["t2i"], ix["c2i"]
        eligible, comp_cols, gr = ix["eligible"], ix["comp_cols"], ix["gr"]
        n_cell, n_treat = ix["n_cell"], ix["n_treat"]
        rows = []
        for hcn in cells:
            hc = c2i[hcn]
            tr, _ = make_splits(eb, {"holdout_cells": [hcn], "test_treatments": {hcn: set(eligible)}})
            ctr = np.array([c2i[c] for c in tr.cell_line]); ttr = np.array([t2i[t] for t in tr.condition])
            yv = tr.growth_rate.to_numpy(float)
            mu, ce, te = als(yv, ctr, ttr, n_cell, n_treat, n_iter=30)
            yimp = mu + ce[:, None] + te[None, :]; yimp[ctr, ttr] = yv
            obs = np.zeros((n_cell, n_treat), bool); obs[ctr, ttr] = True
            nonh = np.array([i for i in range(n_cell) if i != hc])
            test_pairs = [(hc, t2i[c]) for c in eligible]
            add_hc = np.array([yimp[hc, comp_cols[c]].sum() for c in eligible])
            delta = np.array([(yimp[nonh, t2i[c]] - yimp[np.ix_(nonh, comp_cols[c])].sum(1)).mean()
                              for c in eligible])
            out = {"bar1_add": add_hc, "bar4_interaction": add_hc + delta}
            als_test = mu + ce[hc] + te[np.array([t2i[c] for c in eligible])]
            for lam in lams:
                yd, cov = dreg(yimp, obs, test_pairs, lam=lam, holdout_set={hc})
                out[f"bar2_shrink0_lam{lam:g}"] = np.where(cov, yd, als_test)
                yimp_r = yimp.copy()
                for c in eligible:
                    yimp_r[:, t2i[c]] = yimp[:, t2i[c]] - yimp[np.ix_(np.arange(n_cell), comp_cols[c])].sum(1)
                yr, cov3 = dreg(yimp_r, obs, test_pairs, lam=lam, holdout_set={hc})
                out[f"bar3_prior11_lam{lam:g}"] = np.where(cov3, yr + add_hc, add_hc)
            for k, c in enumerate(eligible):
                rows.append({"cell_line": hcn, "condition": c,
                             "drugs": "+".join(d for d, _, _ in ast.literal_eval(c)),
                             "measured": gr[(hcn, c)],
                             **{col: float(v[k]) for col, v in out.items()}})
        return pd.DataFrame(rows)

    def fit_wprior(eb, weights, ordering="alpha", lam=LAM):
        """bar3-style fit with an arbitrary 2-vector `weights` on the (ordered)
        ingredients. ordering="alpha" -> weight[0] on alphabetically-first drug;
        "potent" -> weight[0] on the more-potent drug (more-negative mean
        single-agent growth_rate). weights=[1,1] reproduces bar3 exactly."""
        from rhaister.prepare_sensitivity import make_splits
        from rhaister.train_sensitivity import _als_decompose_scalar as als, _drug_regression as dreg
        ix = build_index(eb)
        cells, t2i, c2i = ix["cells"], ix["t2i"], ix["c2i"]
        eligible, single_cond, gr = ix["eligible"], ix["single_cond"], ix["gr"]
        n_cell, n_treat = ix["n_cell"], ix["n_treat"]
        is_multi = ix["is_multi_drug"]
        if ordering == "potent":
            pot = (eb[~eb["condition"].map(is_multi)]
                   .groupby("condition")["growth_rate"].mean().to_dict())
            key = lambda t: pot[single_cond[(t[0], t[1])]]
        elif ordering == "alpha":
            key = lambda t: t[0]
        else:
            raise ValueError(ordering)
        comp_cols = {c: [t2i[single_cond[(d, dose)]]
                         for d, dose, _u in sorted(ast.literal_eval(c), key=key)]
                     for c in eligible}
        W = np.asarray(weights, float)
        rows = []
        for hcn in cells:
            hc = c2i[hcn]
            tr, _ = make_splits(eb, {"holdout_cells": [hcn], "test_treatments": {hcn: set(eligible)}})
            ctr = np.array([c2i[c] for c in tr.cell_line]); ttr = np.array([t2i[t] for t in tr.condition])
            yv = tr.growth_rate.to_numpy(float)
            mu, ce, te = als(yv, ctr, ttr, n_cell, n_treat, n_iter=30)
            yimp = mu + ce[:, None] + te[None, :]; yimp[ctr, ttr] = yv
            obs = np.zeros((n_cell, n_treat), bool); obs[ctr, ttr] = True
            test_pairs = [(hc, t2i[c]) for c in eligible]
            prior_hc = np.array([(yimp[hc, comp_cols[c]] * W).sum() for c in eligible])
            res = yimp.copy()
            for c in eligible:
                res[:, t2i[c]] = yimp[:, t2i[c]] - (yimp[:, comp_cols[c]] * W).sum(1)
            yr, cov = dreg(res, obs, test_pairs, lam=lam, holdout_set={hc})
            pred = np.where(cov, yr + prior_hc, prior_hc)
            for k, c in enumerate(eligible):
                rows.append({"cell_line": hcn, "condition": c,
                             "pred": float(pred[k]), "measured": float(gr[(hcn, c)])})
        return pd.DataFrame(rows)

    def ab_halfsplit_ceiling(eligible, seeds=AB_SEEDS):
        """Random half-split reliability of growth_rate from the raw h5ad counts.
        Per combo, correlate half-A vs half-B growth_rate across the 52 cells;
        Spearman-Brown -> r_full (upper bound on achievable per-perturbation R^2).
        Returns {cond: r_full} or None if H5_DIR is absent. CAVEAT: a random-cell
        split sees only sampling noise -> r_full is an upper bound."""
        if not os.path.isdir(H5_DIR):
            warnings.warn(f"H5_DIR not found ({H5_DIR}); skipping A/B ceiling.")
            return None
        import h5py
        ctrl = "[('DMSO_TF', 0.0, 'uM')]"

        def cats(f, key):
            c = f[f"obs/{key}/categories"][:]
            return [x.decode() if isinstance(x, bytes) else x for x in c]

        data = []
        for path in sorted(glob.glob(f"{H5_DIR}/*.h5ad")):
            with h5py.File(path, "r") as f:
                cl = cats(f, "cell_line")[np.bincount(f["obs/cell_line/codes"][:]).argmax()]
                dcat = cats(f, "drugname_drugconc")
                dco = f["obs/drugname_drugconc/codes"][:].astype(np.int32)
                data.append((cl, dco, dcat))

        def gr_half(which, masks):
            counts = {}
            for (cl, dco, dcat), mask in zip(data, masks):
                m = mask if which == "A" else ~mask
                nb = np.bincount(dco[m], minlength=len(dcat))
                counts[cl] = {dcat[i]: int(nb[i]) for i in range(len(dcat)) if nb[i] > 0}
            tot = {}
            for cd in counts.values():
                for cond, n in cd.items():
                    tot[cond] = tot.get(cond, 0) + n
            gr = {}
            for cl, cd in counts.items():
                nc = cd.get(ctrl, 0)
                if nc == 0 or tot.get(ctrl, 0) == 0:
                    continue
                pc = nc / tot[ctrl]
                for cond, n in cd.items():
                    if cond != ctrl and n > 0:
                        gr[(cl, cond)] = np.log2((n / tot[cond]) / pc)
            return gr

        per = {c: [] for c in eligible}
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            masks = [rng.random(len(dco)) < 0.5 for (_, dco, _) in data]
            ga, gb = gr_half("A", masks), gr_half("B", masks)
            for c in eligible:
                a, b = [], []
                for (cl, _, _) in data:
                    if (cl, c) in ga and (cl, c) in gb:
                        a.append(ga[(cl, c)]); b.append(gb[(cl, c)])
                if len(a) >= 3:
                    per[c].append(float(np.corrcoef(a, b)[0, 1]))
        r_full = {}
        for c in eligible:
            rh = float(np.mean(per[c])) if per[c] else float("nan")
            r_full[c] = 2 * rh / (1 + rh)
        return r_full

    return ab_halfsplit_ceiling, fit_bars_2x2, fit_bars_all52, fit_wprior


@app.cell
def _(
    CLASS_COLORS,
    SW480,
    ast,
    attach_design_class,
    fig_path,
    fit_wprior,
    np,
    pd,
    per_perturbationr2_score,
    plt,
    r2_score,
    spearmanr,
):
    # ======================================================================
    # Experiment functions.  Each returns its matplotlib Figure (so marimo
    # displays it) and also writes the PNG to disk.
    # ======================================================================
    def exp1_sw480_proof(eb, ab_r_full, build_index):
        """SW480 leave-one-out: bar1 (additivity), bar2 (ships, lam=1), bar3
        lambda sweep. FINDING: on SW480 additivity wins -- bar3 R^2 ~ 0.89 at
        lam in [1,10], vs ~0.66/0.70 for bar1/bar2. Does NOT generalise (exp2)."""
        from rhaister.prepare_sensitivity import make_splits
        from rhaister.train_sensitivity import _als_decompose_scalar as als, _drug_regression as dreg
        ix = build_index(eb)
        t2i, c2i, gr = ix["t2i"], ix["c2i"], ix["gr"]
        eligible, comp_cols = ix["eligible"], ix["comp_cols"]
        n_cell, n_treat = ix["n_cell"], ix["n_treat"]
        tr, _te = make_splits(eb, {"holdout_cells": [SW480], "test_treatments": {SW480: set(eligible)}})
        ctr = np.array([c2i[c] for c in tr.cell_line]); ttr = np.array([t2i[t] for t in tr.condition])
        mu, ce, teff = als(tr.growth_rate.to_numpy(float), ctr, ttr, n_cell, n_treat, n_iter=30)
        yimp = mu + ce[:, None] + teff[None, :]; yimp[ctr, ttr] = tr.growth_rate.to_numpy(float)
        obs = np.zeros((n_cell, n_treat), bool); obs[ctr, ttr] = True
        hc = c2i[SW480]; combos = list(eligible)
        measured = np.array([gr[(SW480, c)] for c in combos])
        bar1 = np.array([yimp[hc, comp_cols[c]].sum() for c in combos])
        yd, cov = dreg(yimp, obs, [(hc, t2i[c]) for c in combos], lam=1.0, holdout_set={hc})
        bar2 = np.where(cov, yd, mu + ce[hc] + teff[np.array([t2i[c] for c in combos])])
        nonh = np.array([i for i in range(n_cell) if i != hc])
        Dx = sorted({t2i[t] for t in tr.loc[tr.cell_line == SW480, "condition"]})
        pos = {t: j for j, t in enumerate(Dx)}
        X = yimp[np.ix_(nonh, Dx)]; xh = yimp[hc, Dx]; XtX = X.T @ X
        comp_local = [[pos[g] for g in comp_cols[c]] for c in combos]
        XtY = X.T @ yimp[np.ix_(nonh, [t2i[c] for c in combos])]
        sweep = []
        for lam in [1e-2, 1e-1, 1.0, 3.0, 10.0, 100.0, 1e4, 1e8]:
            M = XtX + lam * np.eye(len(Dx)); pred = np.empty(len(combos))
            for k in range(len(combos)):
                wp = np.zeros(len(Dx))
                for ci in comp_local[k]:
                    wp[ci] = 1.0
                pred[k] = xh @ np.linalg.solve(M, XtY[:, k] + lam * wp)
            sweep.append({"lambda": lam, "R2": round(r2_score(measured, pred), 4),
                          "Pearson": round(float(np.corrcoef(measured, pred)[0, 1]), 4)})
        return {"bar1": {"R2": round(r2_score(measured, bar1), 4)},
                "bar2": {"R2": round(r2_score(measured, bar2), 4)},
                "bar3_sweep": pd.DataFrame(sweep), "n_test": len(combos)}

    def exp2_fig1a_panel(bars52, ab_r_full):
        """Four bars across 52 cells, pooled + per-cell. FINDING: 'Rhaister
        ships' (bar2) wins the panel; additivity (bar1) is at chance; the [1,1]
        bars sit far lower -- the SW480 story inverts."""
        bars = ["bar1", "bar2", "bar3", "bar4"]
        labels = ["add-the-\nsingles", "shrink-0 +int\n(ships)", "[1,1]\nno int", "[1,1]\n+int"]
        r2 = [bars52["pooled"][b]["R2"] for b in bars]
        pe = [bars52["pooled"][b]["Pearson"] for b in bars]
        pcv = [bars52["per_cell_R2"][b] for b in bars]
        swi = bars52["sw480_idx"]
        ceiling = float(np.nanmedian(list(ab_r_full.values()))) if ab_r_full else None
        x = np.arange(4)
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
        w = 0.38
        axA.bar(x - w / 2, r2, w, label="R^2", color="#4C72B0")
        axA.bar(x + w / 2, pe, w, label="Pearson", color="#DD8452")
        if ceiling is not None:
            axA.axhline(ceiling, ls="--", color="gray", lw=1)
            axA.text(0.0, ceiling + 0.015, f"A/B ceiling {ceiling:.2f}", fontsize=8, color="gray")
        axA.axhline(0, color="k", lw=0.6); axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=7.5)
        axA.set_ylabel("pooled score (884 combos)"); axA.set_ylim(-0.15, 1.0)
        axA.legend(fontsize=8, loc="center right")
        axA.set_title(f"Pooled across 52 cells (lambda={bars52['lam']:g})", fontsize=10)
        bp = axB.boxplot(pcv, showfliers=False, positions=x, widths=0.5, patch_artist=True)
        for p in bp["boxes"]:
            p.set_facecolor("#B0C4DE")
        for i in range(4):
            axB.plot(x[i], pcv[i][swi], marker="*", color="red", ms=13, zorder=6)
        axB.axhline(0, color="k", lw=0.6); axB.set_ylim(-1.2, 1.0)
        axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=7.5)
        axB.set_ylabel("per-cell R^2"); axB.set_title("Per-cell spread (red * = SW480)", fontsize=10)
        fig.suptitle("Fig 1a - held-out combo growth rate (leave-one-cell-out, 52 x 17)", fontsize=11)
        fig.tight_layout(); fig.savefig(fig_path("fig1a_bars.png"), dpi=150, bbox_inches="tight")
        return fig

    def exp3_bar3_vs_bar4(bars52, names):
        """Per-cell R^2 scatter bar3 vs bar4, labelling every R^2<0 cell.
        FINDING: points hug the diagonal (intercept ~ no-op); 20/52 negative;
        failure is a wrong direction, not a constant offset. Returns (fig, table)."""
        b3 = np.array(bars52["per_cell_R2"]["bar3"]); b4 = np.array(bars52["per_cell_R2"]["bar4"])
        cvcl = np.array(bars52["cells"]); swi = bars52["sw480_idx"]
        lo, hi = -1.5, 1.0
        b3c, b4c = np.clip(b3, lo, hi), np.clip(b4, lo, hi)
        label = np.array([str(names["cell_name"].get(c, c)) for c in cvcl])
        col = np.where((b3 > 0) & (b4 > 0), "#2a9d8f",
                       np.where((b3 < 0) & (b4 < 0), "#e76f51", "#9aa0a6"))
        fig, ax = plt.subplots(figsize=(7.6, 7.6))
        ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.7)
        ax.plot([lo, hi], [lo, hi], ls="--", color="k", lw=0.8, label="bar 3 = bar 4")
        ax.scatter(b3c, b4c, c=col, s=45, edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter([b3c[swi]], [b4c[swi]], marker="*", s=320, color="red",
                   edgecolor="black", linewidth=0.5, zorder=5, label="SW480")
        neg = np.where((b3 < 0) | (b4 < 0))[0]
        neg = neg[neg != swi]; neg = neg[np.argsort(b4c[neg])]
        gap, last, laby = 0.104, -1e9, []
        for i in neg:
            yv = max(float(b4c[i]), last + gap); laby.append(yv); last = yv
        for i, ly in zip(neg, laby):
            ax.annotate(label[i], (b3c[i], b4c[i]), xytext=(b3c[i] + 0.06, ly),
                        fontsize=6.2, color="#7a2f1e", ha="left", va="center", zorder=6,
                        arrowprops=dict(arrowstyle="-", color="#e76f51", lw=0.4, alpha=0.55))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("bar 3 per-cell R^2   ([1,1], no intercept)")
        ax.set_ylabel("bar 4 per-cell R^2   ([1,1] + intercept)")
        ax.set_title(f"Per-cell R^2: bar3 vs bar4 ({len(neg)} lines R^2<0 in either bar)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(fig_path("fig_bar3_vs_bar4.png"), dpi=150, bbox_inches="tight")
        tbl = pd.DataFrame({"cell_line": [str(names["cell_name"].get(c, c)) for c in cvcl],
                            "cvcl": cvcl, "organ": [str(names["Organ"].get(c, "?")) for c in cvcl],
                            "bar3_R2": b3.round(3), "bar4_R2": b4.round(3)})
        below = tbl[(tbl.bar3_R2 < 0) | (tbl.bar4_R2 < 0)].sort_values("bar4_R2").reset_index(drop=True)
        return fig, below

    def exp4_failing_cell_profile(pred_all52, eb, names):
        """Per-cell bar3 R^2 vs sensitivity / model error / combo variance.
        FINDING (correction): R^2<0 is NOT 'the drug-sensitive lines'
        (sensitivity corr ~0); it is driven by high model RMSE and low per-combo
        variance. Returns (fig, prof)."""
        is_single = lambda c: len(ast.literal_eval(c)) == 1 and not ast.literal_eval(c)[0][0].startswith("DMSO")
        sens = (eb[eb.condition.map(is_single)].groupby("cell_line")["growth_rate"]
                .agg(mean_single_gr="mean", median_single_gr="median",
                     frac_strong_kill=lambda s: (s < -1).mean()))
        prof = []
        for cl, g in pred_all52.groupby("cell_line"):
            m = g["measured"].to_numpy(); pr = g["bar3_prior11_lam10"].to_numpy(); add = g["bar1_add"].to_numpy()
            prof.append({"cell_line": cl, "r2_bar3": r2_score(m, pr), "meas_std": float(m.std()),
                         "rmse": float(np.sqrt(((m - pr) ** 2).mean())),
                         "mean_abs_synergy": float(np.abs(m - add).mean())})
        prof = pd.DataFrame(prof).merge(sens, on="cell_line", how="left")
        prof["sign"] = np.where(prof.r2_bar3 < 0, "R2<0", "R2>=0")
        prof["organ"] = prof.cell_line.map(lambda c: names["Organ"].to_dict().get(c, "unknown"))
        corr_sens = prof[["r2_bar3", "mean_single_gr"]].corr().iloc[0, 1]
        sp_rmse = prof[["r2_bar3", "rmse"]].rank().corr().iloc[0, 1]
        sp_std = prof[["r2_bar3", "meas_std"]].rank().corr().iloc[0, 1]
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.4))
        cmap = prof.sign.map({"R2<0": "#e76f51", "R2>=0": "#2a9d8f"})
        a.scatter(prof.mean_single_gr, prof.r2_bar3, c=cmap, s=55, edgecolor="white")
        a.axhline(0, color="k", lw=0.7); a.axvline(0, color="gray", lw=0.6); a.set_ylim(-6, 1.05)
        a.set_xlabel("mean single-agent growth_rate\n(more negative = more sensitive)")
        a.set_ylabel("bar3 R^2"); a.set_title(f"sensitivity (r={corr_sens:+.2f}, ~0)")
        b.scatter(prof.meas_std, prof.r2_bar3, c=cmap, s=55, edgecolor="white")
        b.axhline(0, color="k", lw=0.7); b.set_ylim(-6, 1.05)
        b.set_xlabel("measured combo std (per cell)"); b.set_title(f"low variance (Spearman {sp_std:+.2f})")
        c.scatter(prof.rmse, prof.r2_bar3, c=cmap, s=55, edgecolor="white")
        c.axhline(0, color="k", lw=0.7); c.set_ylim(-6, 1.05)
        c.set_xlabel("model RMSE (per cell)"); c.set_title(f"high error (Spearman {sp_rmse:+.2f})")
        fig.tight_layout(); fig.savefig(fig_path("fig_failing_cell_profile.png"), dpi=150, bbox_inches="tight")
        return fig, prof

    def exp5_bar3_vs_potency_percell(prof):
        """Per-cell bar3 R^2 vs single-agent potency. FINDING: flat cloud
        (Spearman ~ -0.17) -- potency does not predict where bar3 fails."""
        x = prof.mean_single_gr.to_numpy(); y = np.clip(prof.r2_bar3.to_numpy(), -2.0, 1.05)
        rho = spearmanr(x, prof.r2_bar3).correlation
        col = np.where(prof.r2_bar3 < 0, "#e76f51", "#2a9d8f")
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.6, ls=":")
        ax.scatter(x, y, c=col, s=55, edgecolor="white", zorder=3)
        ax.set_xlabel("single-agent drug potency = mean single-agent growth_rate\n(left = more potent)")
        ax.set_ylabel("bar 3 per-cell R^2")
        ax.set_title(f"Bar 3 vs drug potency (n=52) -- no relationship (Spearman {rho:+.2f})", fontsize=10)
        fig.tight_layout(); fig.savefig(fig_path("fig_bar3_vs_potency.png"), dpi=150, bbox_inches="tight")
        return fig

    def _scatter_perturb(df, title, ylabel, path, ylo=-1.9):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.6, ls=":")
        ax.scatter(df.potency, df.r2, c=np.where(df.r2 < 0, "#e76f51", "#2a9d8f"),
                   s=80, edgecolor="white", zorder=3)
        ax.set_xlabel("perturbation potency = mean measured growth_rate across 52 cells\n(left = more potent)")
        ax.set_ylabel(ylabel); ax.set_ylim(min(ylo, df.r2.min() - 0.1), 1.05); ax.set_title(title, fontsize=10)
        fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    def _designclass_twopanel(pred_all52, dc, left_col, left_ttl, path, suptitle, ylo=-2.9):
        rows = []
        for cond, g in pred_all52.groupby("condition"):
            m = g["measured"].to_numpy()
            rows.append({"condition": cond, "drugs": g["drugs"].iloc[0], "potency": float(m.mean()),
                         "left": r2_score(m, g[left_col].to_numpy()),
                         "bar3": r2_score(m, g["bar3_prior11_lam10"].to_numpy())})
        d = attach_design_class(pd.DataFrame(rows), dc)
        order = list(d.design_class.value_counts().index)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
        for ax, col, ttl in [(axL, "left", left_ttl), (axR, "bar3", "bar 3: [1,1] additivity-prior ridge")]:
            for cls in order:
                s = d[d.design_class == cls]
                ax.scatter(s.potency, s[col], s=110, color=CLASS_COLORS.get(cls, "#999"),
                           edgecolor="white", linewidth=0.8, label=cls, zorder=3)
            ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.6, ls=":")
            ax.set_title(ttl, fontsize=10)
            ax.set_xlabel("potency = mean growth_rate across 52 cells (left = more potent)")
        axL.set_ylabel("per-perturbation R^2"); axL.set_ylim(ylo, 1.08)
        h, l = axL.get_legend_handles_labels()
        fig.legend(h, l, title="combo design class", loc="upper center", ncol=len(order),
                   fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(suptitle, fontsize=11, y=1.06)
        fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    def exp6_perturbationr2_score(pred_all52, per_perturbation_r2):
        """Per-PERTURBATION R^2 vs potency for bar3 and bar2. FINDING: bar2
        keeps ALL 17 combos R^2>=0; bar3 fails on the MAPK combos; potency does
        not predict R^2. Returns (fig_bar3, fig_bar2)."""
        b3 = per_perturbationr2_score(pred_all52, "bar3_prior11_lam10")
        b2 = per_perturbationr2_score(pred_all52, "bar2_shrink0_lam10")
        f3 = _scatter_perturb(b3, f"Bar 3 R^2 per perturbation (Spearman {spearmanr(b3.potency, b3.r2).correlation:+.2f})",
                              "per-perturbation R^2 (bar3, [1,1])", fig_path("fig_bar3_r2_per_perturbation.png"))
        f2 = _scatter_perturb(b2, f"Bar 2 (Rhaister ships) R^2 per perturbation -- all {(b2.r2 >= 0).sum()}/17 >= 0",
                              "per-perturbation R^2 (bar2, ships)", fig_path("fig_bar2_r2_per_perturbation.png"), ylo=-0.1)
        return f3, f2

    def exp7_designclass(pred_all52, dc):
        """Per-perturbation R^2 coloured by design class (bar1 vs bar3, bar2 vs
        bar3). FINDING: additivity fails on the synergy-by-design MAPK classes;
        bar2 erases the class structure. Returns (fig_bar1v3, fig_bar2v3)."""
        f1 = _designclass_twopanel(pred_all52, dc, "bar1_add", "bar 1: add-the-singles (pure additivity)",
                                   fig_path("fig_bar1_bar3_by_designclass.png"),
                                   "Per-perturbation R^2 vs potency by design class (bar1 vs bar3)")
        f2 = _designclass_twopanel(pred_all52, dc, "bar2_shrink0_lam10", "bar 2: 'Rhaister ships' (shrink-0 + intercept)",
                                   fig_path("fig_bar2_bar3_by_designclass.png"),
                                   "Per-perturbation R^2 vs potency by design class (bar2 vs bar3)", ylo=-1.9)
        return f1, f2

    PRIORS = [("[1, 1]  (= bar3 ref)", [1, 1]), ("[1, -1]", [1, -1]), ("[-1, 1]", [-1, 1]),
              ("[0.5, 0.5]", [0.5, 0.5]), ("[0.75, 0.25]", [0.75, 0.25]), ("[0.25, 0.75]", [0.25, 0.75])]
    PANEL_ORDER = ["[1, 1]  (= bar3 ref)", "[0.5, 0.5]", "[0.75, 0.25]", "[0.25, 0.75]", "[1, -1]", "[-1, 1]"]

    def exp8_wprior_sweep(eb, dc, ordering, per_perturbation_r2, build_index):
        """Sweep the additivity prior W (6 settings) at a chosen ingredient
        `ordering`. FINDING: every sum-1 prior (mean, not sum) beats sum-2
        [1,1]; the asymmetric trend under 'alpha' ordering VANISHES under
        'potent' ordering -- it is not a potency effect. Returns (fig, tidy)."""
        tidy = []
        for lbl, w in PRIORS:
            pf = fit_wprior(eb, w, ordering=ordering)
            tidy.append(per_perturbationr2_score(pf, "pred").assign(prior=lbl))
        tidy = attach_design_class(pd.concat(tidy, ignore_index=True), dc)
        order = list(tidy.design_class.value_counts().index)
        fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, prior in zip(axes, PANEL_ORDER):
            d = tidy[tidy.prior == prior]
            for cls in order:
                s = d[d.design_class == cls]
                ax.scatter(s.potency, s.r2, s=95, color=CLASS_COLORS.get(cls, "#999"),
                           edgecolor="white", linewidth=0.7, label=cls, zorder=3)
            ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.6, ls=":")
            ax.set_title(f"W_prior = {prior}\nmedian R^2 = {d.r2.median():+.2f} | R^2<0: {(d.r2 < 0).sum()}/17",
                         fontsize=9.5)
        for i in (0, 3):
            axes[i].set_ylabel("per-perturbation R^2")
        for i in (3, 4, 5):
            axes[i].set_xlabel("potency = mean growth_rate (left = more potent)")
        axes[0].set_ylim(-1.9, 1.08)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, title="combo design class", loc="upper center", ncol=len(order),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.005))
        cap = ("[alphabetically-first drug, second]" if ordering == "alpha"
               else "[more-potent partner, less-potent]")
        fig.suptitle(f"bar3-style fit under different additivity W_priors (n=17, lam=10)\nweights = {cap}",
                     fontsize=11, y=1.06)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        suffix = "" if ordering == "alpha" else "_potentfirst"
        fig.savefig(fig_path(f"fig_wprior_sweep{suffix}_by_designclass.png"), dpi=150, bbox_inches="tight")
        return fig, tidy

    def exp9_baselines(pred_all52, dc, ab_r_full):
        """bar1, bar2, and the per-combo A/B reliability ceiling, by design
        class. FINDING: the lowest-reliability combos ARE the MAPK class, so
        part of bar1/bar3's failure is a real noise floor -- but bar2 stays
        positive above it. Returns fig."""
        rows = []
        for cond, g in pred_all52.groupby("condition"):
            m = g["measured"].to_numpy()
            rows.append({"condition": cond, "drugs": g["drugs"].iloc[0], "potency": float(m.mean()),
                         "r2_bar1": r2_score(m, g["bar1_add"].to_numpy()),
                         "r2_bar2": r2_score(m, g["bar2_shrink0_lam10"].to_numpy()),
                         "ab_r_full": (ab_r_full or {}).get(cond, np.nan)})
        d = attach_design_class(pd.DataFrame(rows), dc)
        order = list(d.design_class.value_counts().index)
        npanel = 3 if ab_r_full else 2
        fig, axes = plt.subplots(1, npanel, figsize=(5.5 * npanel, 5.6))
        panels = [("r2_bar1", "bar 1: add-the-singles (additivity)", (-2.9, 1.08), "per-perturbation R^2"),
                  ("r2_bar2", "bar 2: 'Rhaister ships'", (-2.9, 1.08), "per-perturbation R^2")]
        if ab_r_full:
            panels.append(("ab_r_full", "A/B split-half ceiling (reliability)", (0.5, 1.0), "Spearman-Brown r_full"))
        for ax, (col, ttl, ylim, ylab) in zip(np.atleast_1d(axes), panels):
            for cls in order:
                s = d[d.design_class == cls]
                ax.scatter(s.potency, s[col], s=100, color=CLASS_COLORS.get(cls, "#999"),
                           edgecolor="white", linewidth=0.8, label=cls, zorder=3)
            ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.6, ls=":")
            med = d[col].median(); neg = int((d[col] < 0).sum())
            extra = f" | R^2<0: {neg}/17" if col.startswith("r2") else ""
            ax.set_title(f"{ttl}\nmedian {med:+.2f}{extra}", fontsize=9.5)
            ax.set_xlabel("potency = mean growth_rate (left = more potent)"); ax.set_ylim(*ylim); ax.set_ylabel(ylab)
        h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(h, l, title="combo design class", loc="upper center", ncol=len(order),
                   fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.04))
        fig.suptitle("Baseline references per perturbation (bar1, bar2, A/B ceiling) — n=17", fontsize=11.5, y=1.12)
        fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(fig_path("fig_baselines_by_designclass.png"), dpi=150, bbox_inches="tight")
        return fig

    return (
        exp1_sw480_proof,
        exp2_fig1a_panel,
        exp3_bar3_vs_bar4,
        exp4_failing_cell_profile,
        exp5_bar3_vs_potency_percell,
        exp6_perturbationr2_score,
        exp7_designclass,
        exp8_wprior_sweep,
        exp9_baselines,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Setup — load data and run the leave-one-cell-out fits (a few minutes)
    """)
    return


@app.cell
def _(
    LAM,
    ab_halfsplit_ceiling,
    build_index,
    fit_bars_2x2,
    fit_bars_all52,
    load_design_classes,
    load_emeraldbay,
    load_metadata,
    mo,
    name_lookup,
):
    eb = load_emeraldbay()
    cell_line_md, _summary = load_metadata()
    names = name_lookup(cell_line_md)
    dc = load_design_classes()
    _ix = build_index(eb)
    eligible = _ix["eligible"]
    ab_r_full = ab_halfsplit_ceiling(eligible)
    bars52 = fit_bars_2x2(eb, lam=LAM)
    pred_all52 = fit_bars_all52(eb, lams=(1.0, 10.0))
    mo.md(
        f"Loaded **{len(eb)}** rows, **{len(eligible)}** eligible combos. "
        f"A/B ceiling: **{'computed' if ab_r_full else 'skipped (h5ads unavailable)'}**. "
        f"Fits ready: `bars52` (2x2 ridge) and `pred_all52` (_drug_regression)."
    )
    return ab_r_full, bars52, dc, eb, names, pred_all52


@app.cell
def _(dc):
    dc
    return


@app.cell
def _(dc):
    dc['design_class'].value_counts()
    return


@app.cell
def _(dc):
    dc[dc['design_class'] == "horizontal bypass"]
    return


@app.cell
def _(dc):
    dc[dc['design_class'] == "same-node redundancy"]
    return


@app.cell
def _(eb):
    tucatinib = eb['condition'].str.contains('Tucatinib')
    eb[tucatinib]
    return


@app.cell
def _(eb):
    adagrasib = eb['condition'].str.contains('Adagrasib')
    eb[adagrasib]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exp 1 — SW480 proof of concept
    Hold out SW480 only; predict its 17 eligible combos. **Finding:** on
    SW480 additivity is excellent — bar3 reaches R² ≈ 0.89 at λ∈[1,10], far
    above bar1/bar2 (~0.66/0.70). This clean story does **not** generalise.
    """)
    return


@app.cell
def _(ab_r_full, build_index, eb, exp1_sw480_proof, mo):
    _r1 = exp1_sw480_proof(eb, ab_r_full, build_index)
    mo.vstack([
        mo.md(f"**SW480:** bar1 R² = {_r1['bar1']['R2']}, bar2 R² = {_r1['bar2']['R2']}. "
              f"bar3 shrink-to-[1,1] λ-sweep (n={_r1['n_test']}):"),
        _r1["bar3_sweep"],
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exp 2 — Fig 1a: four bars across all 52 cell lines
    **Finding:** the SW480 story inverts. "Rhaister ships" (bar2) wins the
    panel; add-the-singles (bar1) is at chance; the additivity-anchored bars
    3/4 sit far lower.
    """)
    return


@app.cell
def _(ab_r_full, bars52, exp2_fig1a_panel):
    exp2_fig1a_panel(bars52, ab_r_full)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exp 3 — bar3 vs bar4 per-cell scatter
    **Finding:** points hug the diagonal → the intercept (bar4) is ~a no-op;
    20/52 cells are negative in both bars; the failure is a wrong direction
    (assay saturation), not a constant offset. Below-zero cells listed.
    """)
    return


@app.cell
def _(bars52, exp3_bar3_vs_bar4, mo, names):
    _fig3, _below = exp3_bar3_vs_bar4(bars52, names)
    mo.vstack([_fig3, mo.md(f"**{len(_below)}** cell lines with R²<0 in either bar:"), _below])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exp 4 — why do those cells fail?
    **Finding (correction to the original prose):** the R²<0 cells are **not**
    "the drug-sensitive lines" — correlation of bar3 R² with single-agent
    sensitivity is ≈ 0. R²<0 is driven by **high model RMSE** (Spearman ≈ −0.46)
    and **low per-combo variance** (Spearman ≈ +0.45): a low-variance
    denominator plus real error, not sensitivity.
    """)
    return


@app.cell
def _(eb, exp4_failing_cell_profile, names, pred_all52):
    _fig4, prof = exp4_failing_cell_profile(pred_all52, eb, names)
    _fig4
    return (prof,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Exp 5 — per-cell bar3 R² vs single-agent potency
    **Finding:** flat cloud (Spearman ≈ −0.17) — potency does not predict
    where bar3 fails, consistent with Exp 4.
    """)
    return


@app.cell
def _(exp5_bar3_vs_potency_percell, prof):
    exp5_bar3_vs_potency_percell(prof)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exp 6 — per-PERTURBATION R² vs potency (bar3 and bar2)
    Flip the aggregation: one dot per combo, R² across the 52 cells.
    **Finding:** bar2 keeps every combo R²≥0 (0.06–0.77); bar3 fails on the
    MAPK synergy combos; potency does not predict R².
    """)
    return


@app.cell
def _(exp6_perturbationr2_score, mo, per_perturbation_r2, pred_all52):
    _f6a, _f6b = exp6_perturbationr2_score(pred_all52, per_perturbation_r2)
    mo.vstack([_f6a, _f6b])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exp 7 — per-perturbation R² coloured by design class
    **Finding:** the same-process (antimetabolite) combos are reliably
    additive; the synergy-by-design MAPK classes (horizontal bypass, vertical
    blockade) are where bar1/bar3 fail; bar2 lifts them all positive.
    """)
    return


@app.cell
def _(dc, exp7_designclass, mo, pred_all52):
    _f7a, _f7b = exp7_designclass(pred_all52, dc)
    mo.vstack([_f7a, _f7b])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exp 8 — W_prior sweep
    Sweep the additivity prior W ∈ {[1,1], [1,-1], [-1,1], [.5,.5], [.75,.25],
    [.25,.75]} at two ingredient orderings. **Finding:** every sum≈1 prior
    (mean, not sum) beats the sum-2 [1,1] bar3 (median R² ~0.55 vs 0.34). The
    asymmetric trend visible under **alphabetical** ordering **vanishes**
    under **potent-first** ordering — so it is *not* a potency effect.
    Antagonistic priors ([1,-1], [-1,1]) are worst.
    """)
    return


@app.cell
def _(build_index, dc, eb, exp8_wprior_sweep, per_perturbation_r2):
    # ordering by alphabetical drug name
    _f8_alpha, _tidy_alpha = exp8_wprior_sweep(eb, dc, "alpha", per_perturbation_r2, build_index)
    _f8_alpha
    return


@app.cell
def _(build_index, dc, eb, exp8_wprior_sweep, per_perturbation_r2):
    # ordering by single-agent potency (weight[0] on the more-potent partner)
    _f8_potent, _tidy_potent = exp8_wprior_sweep(eb, dc, "potent", per_perturbation_r2, build_index)
    _f8_potent
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exp 9 — baselines (bar1, bar2, A/B ceiling) by design class
    **Finding:** the lowest-reliability combos (A/B r_full ~0.66–0.83) are the
    horizontal-bypass MAPK class, so part of bar1/bar3's failure is a real
    noise floor — but bar2 still stays positive above it.
    """)
    return


@app.cell
def _(ab_r_full, dc, exp9_baselines, pred_all52):
    exp9_baselines(pred_all52, dc, ab_r_full)
    return


if __name__ == "__main__":
    app.run()
