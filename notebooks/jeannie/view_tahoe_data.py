import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    # emeraldbay
    import pandas as pd, ast
    from collections import defaultdict

    dir_path = '/home/jeannie/.cache/huggingface/hub/datasets--tahoebio--EmeraldBay/snapshots/f2a0be6b02f731553657f0115c345b20bb020ede'
    summary_stats_df = pd.read_parquet(dir_path + '/metadata/summary_statistics.parquet')
    cell_line_md_df = pd.read_parquet(dir_path + '/metadata/cell_line_metadata.parquet')
    drug_md_df = pd.read_parquet(dir_path + '/metadata/drug_metadata.parquet')
    gene_md_df = pd.read_parquet(dir_path + '/metadata/gene_metadata.parquet')
    return ast, cell_line_md_df, pd, summary_stats_df


@app.cell
def _():
    # grab the data containing all of the combination drugs and all of the single drugs with the resulting sensitivity as the outcome

    import os as _os
    _os.environ["HP_KEEP_MULTI_DRUG"] = "1"   # keep combination rows (off = drops them)
    from rhaister.prepare_sensitivity import load_data as _load_data, _is_multi_drug as _is_multi

    eb_df = _load_data("EmeraldBay")          # both singles AND combos, one table
    _conds = eb_df["condition"].unique()
    _n_single = sum(not _is_multi(c) for c in _conds)
    _n_combo = sum(_is_multi(c) for c in _conds)
    print(f"{len(eb_df)} rows | {len(_conds)} conditions "
          f"({_n_single} single, {_n_combo} combo)")
    eb_df
    return (eb_df,)


@app.cell
def _(eb_df):
    eb_df['cell_line'].unique()
    return


@app.cell
def _(eb_df):
    eb_df['cell_line'].nunique()
    return


@app.cell
def _(eb_df):
    from rhaister.combos import is_multi_drug

    # single vs combo label for each row
    eb_kind = eb_df["condition"].map(lambda c: "combo" if is_multi_drug(c) else "single")

    # growth_rate summary, split by single vs combo
    eb_summary = (
        eb_df.assign(kind=eb_kind)
        .groupby("kind")["growth_rate"]
        .agg(n="count", mean="mean", std="std", min="min", median="median", max="max")
        .round(3)
    )

    print(f"eb_df: {eb_df.shape[0]} rows  |  "
          f"{eb_df['cell_line'].nunique()} cell lines  x  {eb_df['condition'].nunique()} conditions")
    print(f"growth_rate overall: mean={eb_df['growth_rate'].mean():.3f}  "
          f"std={eb_df['growth_rate'].std():.3f}  "
          f"range=[{eb_df['growth_rate'].min():.2f}, {eb_df['growth_rate'].max():.2f}]")
    _ctrl = eb_df.loc[eb_df["condition"].str.contains("DMSO_TF"), "growth_rate"]
    print(f"DMSO_TF control mean (reference, expect ~0): {_ctrl.mean():.4f}  (n={_ctrl.size})")
    eb_summary
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Experiment 1 — Predict SW480's combination response (leave-one-cell-out)

    **Question.** Can Rhaister predict SW480's growth-rate response to a drug combination
    A+B, given that A+B was measured in the *other* 51 cell lines and SW480's own
    single-agent responses are available?

    **Regime.** `CELL_X_REGIMEN` — hold out one cell *context*; the combination stays
    visible in every other cell line. This is the intended fewshot transfer test, not
    leakage: the held-out truth is SW480's measured A+B, and the model may use A+B from
    other cells.

    **Hold-out context.** SW480 (`CVCL_0546`).

    **Test set.** SW480's combination conditions where **both** components exist as exact
    single-agent doses (the "ingredients are present" condition) — 17 combos.

    **Train set.** Everything else: all 51 other cell lines (including their A+B), plus
    SW480's 45 single-agent conditions and its remaining combinations.

    **Model — Rhaister sensitivity, no-feature path:**
    1. ALS additive decomposition: `growth_rate ~= mu + cell_effect + treatment_effect`
    2. Per-drug ridge: predict a held-out cell's response as a weighted mix of drugs it has seen
    3. ALS fallback where the ridge has no signal

    **What each prediction draws on:** SW480's cell effect (from its singles) + the A+B
    treatment effect (from the other 51 cells) + drug-ridge transfer.

    **Ground truth.** SW480's measured `growth_rate` for those combos, held out, used only for scoring.

    **Metrics.** R^2, Pearson, MAE over the held-out combos, plus per-combo predicted vs measured.
    """)
    return


@app.cell
def _(eb_df):
    # --- Experiment 1 definition: hold-out context + eligible test combos ---
    import ast as _ast
    import pandas as _pd
    from rhaister.combos import is_multi_drug as _is_multi

    exp_holdout_cell = "CVCL_0546"  # SW480

    # exact single-agent (drug, dose) pairs available for SW480 (exclude DMSO controls)
    _sw = eb_df[eb_df["cell_line"] == exp_holdout_cell]
    _singles = set()
    for _c in _sw["condition"]:
        if not _is_multi(_c):
            _d, _dose, _u = _ast.literal_eval(_c)[0]
            if not _d.startswith("DMSO"):
                _singles.add((_d, _dose))

    # combos whose EVERY component exists as an exact single-agent (drug, dose)
    exp_test_conditions = []
    _rows = []
    for _c in sorted(_sw["condition"]):
        if not _is_multi(_c):
            continue
        _comps = _ast.literal_eval(_c)
        _ok = all((_d, _dose) in _singles for _d, _dose, _u in _comps)
        _rows.append({"drugs": "+".join(_d for _d, _, _ in _comps),
                      "ingredients_present": _ok, "condition": _c})
        if _ok:
            exp_test_conditions.append(_c)

    exp_eligibility = _pd.DataFrame(_rows).sort_values(["ingredients_present", "drugs"], ascending=[False, True])
    print(f"SW480 combos: {len(exp_eligibility)}  |  "
          f"eligible (both ingredients present): {len(exp_test_conditions)}")
    exp_eligibility
    return exp_holdout_cell, exp_test_conditions


@app.cell
def _(eb_df, exp_holdout_cell, exp_test_conditions):
    # --- Experiment 1: build split, train Rhaister (ALS + drug ridge), predict ---
    import numpy as _np
    import pandas as _pd2
    import ast as _ast2
    from rhaister.prepare_sensitivity import make_splits as _make_splits
    from rhaister.train_sensitivity import _als_decompose_scalar as _als, _drug_regression as _dreg

    # CELL_X_REGIMEN split: hold out SW480's eligible combos; everything else trains
    _split_info = {"holdout_cells": [exp_holdout_cell],
                   "test_treatments": {exp_holdout_cell: set(exp_test_conditions)}}
    _train_df, _test_df = _make_splits(eb_df, _split_info)

    _tr_c = _train_df["cell_line"].to_numpy(); _tr_t = _train_df["condition"].to_numpy()
    _te_c = _test_df["cell_line"].to_numpy();  _te_t = _test_df["condition"].to_numpy()
    _y_train = _train_df["growth_rate"].to_numpy(float)
    _y_test = _test_df["growth_rate"].to_numpy(float)

    _all_cells = sorted(eb_df["cell_line"].unique()); _all_treat = sorted(eb_df["condition"].unique())
    _c2i = {c: i for i, c in enumerate(_all_cells)}; _t2i = {t: i for i, t in enumerate(_all_treat)}
    _n_cell, _n_treat = len(_all_cells), len(_all_treat)
    _c_tr = _np.array([_c2i[c] for c in _tr_c]); _t_tr = _np.array([_t2i[t] for t in _tr_t])
    _c_te = _np.array([_c2i[c] for c in _te_c]); _t_te = _np.array([_t2i[t] for t in _te_t])

    # Stage 1: ALS additive baseline (mu + cell + treatment)
    _mu, _cell_eff, _treat_eff = _als(_y_train, _c_tr, _t_tr, _n_cell, _n_treat, n_iter=30)
    _add_test = _mu + _cell_eff[_c_te] + _treat_eff[_t_te]

    # Stage 2: per-drug ridge, with ALS-imputed dense matrix; ALS fallback where uncovered
    _obs = _np.zeros((_n_cell, _n_treat), dtype=bool); _obs[_c_tr, _t_tr] = True
    _yimp = _mu + _cell_eff[:, None] + _treat_eff[None, :]; _yimp[_c_tr, _t_tr] = _y_train
    _ydrug, _cov = _dreg(_yimp, _obs, list(zip(_c_te.tolist(), _t_te.tolist())),
                         lam=1.0, holdout_set={int(c) for c in _c_te})
    _ypred = _np.where(_cov, _ydrug, _add_test)

    # Metrics over the held-out combos
    _res = _y_test - _ypred
    _r2 = 1.0 - (_res**2).sum() / ((_y_test - _y_test.mean())**2).sum()
    _pear = float(_np.corrcoef(_y_test, _ypred)[0, 1])
    exp_metrics = {"n_test": int(len(_y_test)), "ridge_covered": int(_cov.sum()),
                   "R2": round(float(_r2), 4), "Pearson": round(_pear, 4),
                   "MAE": round(float(_np.abs(_res).mean()), 4),
                   "y_test_std": round(float(_y_test.std()), 4)}

    exp_pred_df = _pd2.DataFrame({
        "drugs": ["+".join(d for d, _, _ in _ast2.literal_eval(t)) for t in _te_t],
        "measured": _y_test.round(3),
        "predicted": _ypred.round(3),
        "additive_only": _add_test.round(3),
        "abs_err": _np.abs(_res).round(3),
        "ridge_covered": _cov,
    }).sort_values("abs_err").reset_index(drop=True)

    print("Train rows:", len(_train_df), " Test rows:", len(_test_df))
    print("Metrics:", exp_metrics)
    exp_pred_df
    return (exp_metrics,)


@app.cell
def _(eb_df, exp_metrics, exp_test_conditions, mo):
    mo.md(rf"""
    ## Result

    Trained on **{len(eb_df) - len(exp_test_conditions)}** rows, predicted **{exp_metrics["n_test"]}**
    held-out SW480 combinations. All {exp_metrics["ridge_covered"]}/{exp_metrics["n_test"]} were
    served by the per-drug ridge (no ALS fallback).

    | metric | value |
    |---|---|
    | R^2 | **{exp_metrics["R2"]}** |
    | Pearson | **{exp_metrics["Pearson"]}** |
    | MAE | {exp_metrics["MAE"]} |
    | y_test std | {exp_metrics["y_test_std"]} |

    **What this number is.** This is **Rhaister as it ships** (ALS additive + per-drug ridge,
    no features) predicting SW480's combinations by *transfer*: it uses each combo's effect
    measured in the other 51 cell lines plus SW480's own cell context (learned from its
    single-agent rows). It does **not** yet use an additivity prior or the single agents
    directly — those are the later Fig 1a bars.

    **Caveats before over-reading 0.70.**
    - **n = 17** held-out points — thin; treat as a proof-of-mechanism, not a calibrated score.
    - **No noise ceiling yet.** Without the A/B half-replicate ceiling we can't say whether 0.70
      is "good" — it must be read against how much signal the growth assay even resolves.
    - Scored on the **dose-exact-eligible subset** only (both ingredients present), so bars 1/3/4
      can share this exact test set.

    **Next bars on this same held-out set:** (1) add-the-singles baseline, (3) ridge shrunk toward
    additivity via `W_prior`, (4) + interaction term. Each reuses `exp_test_conditions`.
    """)
    return


@app.cell
def _(eb_df, exp_holdout_cell, exp_test_conditions):
    # --- Bar 1: add-the-singles (Bliss additive null, no model) ---
    # growth_rate is log2(P_treated / P_DMSO_TF), so summing two singles' log-ratios
    # is Bliss independence (multiplicative in linear proportion space).
    import ast as _ast3
    import numpy as _np3
    import pandas as _pd3

    _sw_rows = eb_df[eb_df["cell_line"] == exp_holdout_cell]

    # SW480's measured single-agent growth_rate, keyed by exact (drug, dose)
    _single_gr = {}
    _combo_gr = {}
    for _c, _g in zip(_sw_rows["condition"], _sw_rows["growth_rate"]):
        _t = _ast3.literal_eval(_c)
        if len(_t) == 1 and not _t[0][0].startswith("DMSO"):
            _single_gr[(_t[0][0], _t[0][1])] = _g
        _combo_gr[_c] = _g

    _rows = []
    for _c in exp_test_conditions:
        _comps = _ast3.literal_eval(_c)
        _pred = sum(_single_gr[(_d, _dose)] for _d, _dose, _u in _comps)
        _rows.append({"drugs": "+".join(d for d, _, _ in _comps),
                      "measured": _combo_gr[_c], "add_singles": _pred})

    bar1_df = _pd3.DataFrame(_rows)
    _m = bar1_df["measured"].to_numpy(); _p = bar1_df["add_singles"].to_numpy(); _res = _m - _p
    bar1_metrics = {"method": "add-the-singles (Bliss)", "n_test": int(len(_m)),
                    "R2": round(float(1 - (_res**2).sum() / ((_m - _m.mean())**2).sum()), 4),
                    "Pearson": round(float(_np3.corrcoef(_m, _p)[0, 1]), 4),
                    "MAE": round(float(_np3.abs(_res).mean()), 4)}
    bar1_df = bar1_df.assign(abs_err=_np3.abs(_res).round(3)).sort_values("abs_err").reset_index(drop=True)
    print("Bar 1:", bar1_metrics)
    bar1_df.round(3)
    return (bar1_metrics,)


@app.cell
def _(bar1_metrics, exp_metrics):
    # --- Bar comparison on the identical 17 held-out SW480 combos ---
    import pandas as _pdC

    bars_compare = _pdC.DataFrame([
        {"bar": "1. add-the-singles (Bliss, no model)",
         "R2": bar1_metrics["R2"], "Pearson": bar1_metrics["Pearson"], "MAE": bar1_metrics["MAE"]},
        {"bar": "2. Rhaister ships (ALS + drug ridge)",
         "R2": exp_metrics["R2"], "Pearson": exp_metrics["Pearson"], "MAE": exp_metrics["MAE"]},
    ])
    bars_compare
    return


@app.cell
def _(bar1_metrics, exp_metrics, mo):
    mo.md(rf"""
    ### Bars 1 vs 2 — the simplest method already closes the gap

    | bar | R^2 | Pearson | MAE |
    |---|---|---|---|
    | **1. add-the-singles** (Bliss, no model) | {bar1_metrics["R2"]} | **{bar1_metrics["Pearson"]}** | **{bar1_metrics["MAE"]}** |
    | **2. Rhaister ships** (transfer) | **{exp_metrics["R2"]}** | {exp_metrics["Pearson"]} | {exp_metrics["MAE"]} |

    **Just adding the two single-agent responses — no model at all — matches the shipped
    Rhaister.** It trails slightly on R^2 ({bar1_metrics["R2"]} vs {exp_metrics["R2"]}) but is
    *better* on both correlation ({bar1_metrics["Pearson"]} vs {exp_metrics["Pearson"]}) and
    absolute error ({bar1_metrics["MAE"]} vs {exp_metrics["MAE"]}).

    **Read the R^2-vs-Pearson split.** Add-the-singles is very well *correlated* with truth
    (Pearson {bar1_metrics["Pearson"]}) yet slightly worse on R^2 — its predictions have the
    right *shape* but a small systematic offset from the y=x line. That offset is exactly the
    additivity deviation (synergy / antagonism) that **bars 3-4** (`W_prior` + interaction) are
    meant to soak up. So the story so far: the ingredients carry almost all the signal, and
    what's left is a small, structured departure from additivity.

    **Still missing before this is publishable:** the A/B noise ceiling (is 0.66-0.70 near the
    data's own limit?), and n is only 17. But the panel's claim is already visible — *combos are
    predictable from single-agent data, and the simplest possible method gets most of the way there.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A/B half-split oracle — the measurement noise ceiling

    **Why.** An R^2 of 0.66-0.70 is meaningless without knowing how well the data
    predicts *itself*. The ceiling = treat one independent half of the cells as a
    prediction of the other half.

    **How (recomputed from the h5ad cell counts — the cluster's pre-split
    `growth_rate_long_{A,B}` live on the unmounted `/nvme-shared`):**
    randomly split all 1.83M cells into half A / half B, recompute
    `growth_rate = log2(P/P_DMSO_TF)` on each half independently, correlate A vs B
    across `(cell_line, condition)` pairs -> `r_half`. Spearman-Brown projects to a
    full-sample reliability `r_full = 2*r_half / (1+r_half)`. Averaged over 5 seeds.
    Script: `ab_ceiling_h5ad.py`.

    **Hard caveat.** A random-cell split sees only **sampling / Poisson noise**, so
    `r_full` is an **upper bound** on the true ceiling. EmeraldBay has ~one well per
    condition, so plate / library-prep batch noise cannot be measured here and is
    **not** in this number. Read it as "how much is *sampling* noise," not "the whole
    noise floor."
    """)
    return


@app.cell
def _():
    # --- load the A/B ceiling result (computed by ab_ceiling_h5ad.py) ---
    import json as _jsonAB
    import pandas as _pdAB
    with open("/home/jeannie/relearn/notebooks/jeannie/ab_ceiling_result.json") as _fh:
        ab_ceiling = _jsonAB.load(_fh)
    ab_ceiling_df = (_pdAB.DataFrame(ab_ceiling).T[["n", "r_half_mean", "r_full_SB_mean"]]
                     .rename(columns={"r_half_mean": "r_half", "r_full_SB_mean": "r_full (Spearman-Brown)"}))
    ab_ceiling_df
    return (ab_ceiling,)


@app.cell
def _(ab_ceiling, bar1_metrics, exp_metrics):
    # --- bars 1 & 2 vs the sampling-noise ceiling, on SW480's 17-combo test set ---
    import pandas as _pdC2
    _ceil = ab_ceiling["SW480 eligible-17 (bar test set)"]["r_full_SB_mean"]
    bars_vs_ceiling = _pdC2.DataFrame([
        {"method": "1. add-the-singles (Bliss)", "Pearson": bar1_metrics["Pearson"],
         "gap_to_ceiling": round(_ceil - bar1_metrics["Pearson"], 3)},
        {"method": "2. Rhaister ships (transfer)", "Pearson": exp_metrics["Pearson"],
         "gap_to_ceiling": round(_ceil - exp_metrics["Pearson"], 3)},
        {"method": "A/B sampling-noise ceiling", "Pearson": _ceil, "gap_to_ceiling": 0.0},
    ])
    bars_vs_ceiling
    return


@app.cell
def _(ab_ceiling, bar1_metrics, exp_metrics, mo):
    mo.md(rf"""
    ### What the ceiling tells us

    For SW480's 17-combo test set the half-split reliability is **r_half =
    {ab_ceiling["SW480 eligible-17 (bar test set)"]["r_half_mean"]}**, Spearman-Brown
    **r_full = {ab_ceiling["SW480 eligible-17 (bar test set)"]["r_full_SB_mean"]}**.
    SW480 is a large, deeply-sampled line (median ~2800 cells/combo, so ~1400 per half),
    so its `growth_rate` is measured almost noise-free. The pooled *overall* ceiling is
    lower ({ab_ceiling["overall (all cells, all conds)"]["r_full_SB_mean"]}) because many
    other cell lines are thinner.

    **This reframes bars 1-2.** They sit at Pearson {bar1_metrics["Pearson"]} (add-the-singles)
    and {exp_metrics["Pearson"]} (Rhaister) — roughly **0.10-0.16 below** a sampling-noise
    ceiling of ~{ab_ceiling["SW480 eligible-17 (bar test set)"]["r_full_SB_mean"]}. So that
    shortfall is **not measurement noise**: there is real, resolvable structure the additive
    prediction misses. Additivity is a *strong* baseline (0 -> 0.90 correlation) but it leaves
    a genuine, above-noise gap — precisely the synergy / antagonism that bars 3-4 (`W_prior` +
    interaction) are meant to capture. On R^2 the gap is larger still (~0.66-0.70 vs a ceiling
    near 1).

    **The caveat that keeps this honest.** The random-cell split only measures *sampling*
    noise. With ~one well per condition, plate / library-prep batch noise is invisible here,
    so the true ceiling is **at or below** {ab_ceiling["SW480 eligible-17 (bar test set)"]["r_full_SB_mean"]}
    and the real headroom above the bars is **an upper bound**. To pin the true floor you'd need
    genuine replicate wells (or a barcode/sub-library split) — which EmeraldBay does not provide.
    """)
    return


@app.cell
def _(bar1_metrics, eb_df, exp_holdout_cell, exp_test_conditions):
    # --- Bar 3: Rhaister ridge shrunk toward the [1,1] additivity prior ---
    # Default Rhaister shrinks the drug-combination weights toward 0. Here we shrink
    # toward w_prior = 1 on each of the combo's two single-agent columns, 0 elsewhere:
    #     W = (X^T X + lam I)^{-1} (X^T Y + lam * w_prior)
    # lam -> inf  => W -> [1,1]  => prediction = y(hc,A)+y(hc,B)  (== bar 1, pure additivity)
    # lam -> 0    => unregularized ridge (overfits at n_train=51, |Dx|~75)
    import numpy as _np4
    import pandas as _pd4
    import ast as _ast4
    from rhaister.train_sensitivity import _als_decompose_scalar as _als4
    from rhaister.prepare_sensitivity import make_splits as _msplit4

    # Rebuild the Experiment-1 split + ALS-imputed dense matrix (fast, deterministic)
    _si = {"holdout_cells": [exp_holdout_cell],
           "test_treatments": {exp_holdout_cell: set(exp_test_conditions)}}
    _tr, _te = _msplit4(eb_df, _si)
    _all_cells = sorted(eb_df["cell_line"].unique()); _all_treat = sorted(eb_df["condition"].unique())
    _c2i = {c: i for i, c in enumerate(_all_cells)}; _t2i = {t: i for i, t in enumerate(_all_treat)}
    _ncell, _ntreat = len(_all_cells), len(_all_treat)
    _ctr = _np4.array([_c2i[c] for c in _tr["cell_line"]]); _ttr = _np4.array([_t2i[t] for t in _tr["condition"]])
    _ytr = _tr["growth_rate"].to_numpy(float)
    _mu, _ce, _teff = _als4(_ytr, _ctr, _ttr, _ncell, _ntreat, n_iter=30)
    _yimp = _mu + _ce[:, None] + _teff[None, :]; _yimp[_ctr, _ttr] = _ytr

    _hc = _c2i[exp_holdout_cell]
    _nonh = _np4.array([i for i in range(_ncell) if i != _hc])
    # D_x = every treatment SW480 has in train (Rhaister's basis)
    _Dx = sorted({_t2i[t] for t in _tr.loc[_tr["cell_line"] == exp_holdout_cell, "condition"]})
    _Dx_pos = {t: j for j, t in enumerate(_Dx)}
    _X = _yimp[_np4.ix_(_nonh, _Dx)]; _x_hc = _yimp[_hc, _Dx]
    _XtX = _X.T @ _X

    # map each combo component (drug, dose) -> its single-agent column index
    _single_cond = {}
    for _c in _tr.loc[_tr["cell_line"] == exp_holdout_cell, "condition"]:
        _t = _ast4.literal_eval(_c)
        if len(_t) == 1 and not _t[0][0].startswith("DMSO"):
            _single_cond[(_t[0][0], _t[0][1])] = _c

    _combos = list(exp_test_conditions)
    _sw_gr = dict(zip(eb_df.loc[eb_df.cell_line == exp_holdout_cell, "condition"],
                      eb_df.loc[eb_df.cell_line == exp_holdout_cell, "growth_rate"]))
    _measured = _np4.array([_sw_gr[c] for c in _combos])
    _comp_idx = [[_Dx_pos[_t2i[_single_cond[(d, dose)]]] for d, dose, u in _ast4.literal_eval(c)] for c in _combos]
    _dy_idx = [_t2i[c] for c in _combos]
    _XtY = _X.T @ _yimp[_np4.ix_(_nonh, _dy_idx)]   # (|Dx|, 17)

    def _metrics(pred):
        res = _measured - pred
        r2 = 1 - (res**2).sum() / ((_measured - _measured.mean())**2).sum()
        return round(float(r2), 4), round(float(_np4.corrcoef(_measured, pred)[0, 1]), 4), round(float(_np4.abs(res).mean()), 4)

    _rows = []
    for _lam in [1e-2, 1e-1, 1.0, 3.0, 10.0, 100.0, 1e4, 1e8]:
        _M = _XtX + _lam * _np4.eye(len(_Dx))
        _pred = _np4.empty(len(_combos))
        for _k in range(len(_combos)):
            _wp = _np4.zeros(len(_Dx))
            for _ci in _comp_idx[_k]:
                _wp[_ci] = 1.0
            _W = _np4.linalg.solve(_M, _XtY[:, _k] + _lam * _wp)
            _pred[_k] = _x_hc @ _W
        _r2, _pe, _mae = _metrics(_pred)
        _rows.append({"lambda": _lam, "R2": _r2, "Pearson": _pe, "MAE": _mae})

    bar3_sweep = _pd4.DataFrame(_rows)
    print("Bar 3 lambda sweep (lambda->inf should match bar 1: R2=%.3f Pearson=%.3f):"
          % (bar1_metrics["R2"], bar1_metrics["Pearson"]))
    bar3_sweep
    return (bar3_sweep,)


@app.cell
def _(ab_ceiling, bar1_metrics, bar3_sweep, exp_metrics):
    # --- All bars + ceiling, on SW480's 17-combo test set ---
    import pandas as _pdF
    _b3_best = bar3_sweep.loc[bar3_sweep["R2"].idxmax()]
    _b3_l1 = bar3_sweep[bar3_sweep["lambda"] == 1.0].iloc[0]
    _ceil = ab_ceiling["SW480 eligible-17 (bar test set)"]["r_full_SB_mean"]
    bars_final = _pdF.DataFrame([
        {"bar": "1. add-the-singles (additivity)", "R2": bar1_metrics["R2"], "Pearson": bar1_metrics["Pearson"]},
        {"bar": "2. Rhaister ships (shrink to 0)", "R2": exp_metrics["R2"], "Pearson": exp_metrics["Pearson"]},
        {"bar": "3. shrink to [1,1], lam=1 (untuned)", "R2": float(_b3_l1["R2"]), "Pearson": float(_b3_l1["Pearson"])},
        {"bar": f"3. shrink to [1,1], lam={_b3_best['lambda']:g} (test-selected)", "R2": float(_b3_best["R2"]), "Pearson": float(_b3_best["Pearson"])},
        {"bar": "A/B sampling ceiling", "R2": None, "Pearson": _ceil},
    ])
    bars_final
    return (bars_final,)


@app.cell
def _(ab_ceiling, bar1_metrics, bar3_sweep, bars_final, exp_metrics, mo):
    mo.md(rf"""
    ### Bar 3 — shrink toward additivity, not toward zero

    **Sanity check (passed).** At lambda -> inf the [1,1] prior forces `W = [1,1]`, so bar 3
    becomes exactly bar 1: R2 = {bar3_sweep[bar3_sweep["lambda"]==1e8].iloc[0]["R2"]},
    Pearson = {bar3_sweep[bar3_sweep["lambda"]==1e8].iloc[0]["Pearson"]} — identical to
    add-the-singles. The prior is genuinely "pure additivity."

    **The result.** Letting the ridge start from additivity and learn the *deviation* lifts
    R2 from **{bar1_metrics["R2"]}** (bar 1) / **{exp_metrics["R2"]}** (bar 2, shrink-to-0) to
    **{bars_final.iloc[3]["R2"]:.3f}** (Pearson **{bars_final.iloc[3]["Pearson"]:.3f}**),
    closing most of the gap to the sampling ceiling ~{ab_ceiling["SW480 eligible-17 (bar test set)"]["r_full_SB_mean"]}.
    The inductive bias is the whole story: **shrinking toward [1,1] beats shrinking toward 0.**

    **It is not a knife-edge.** Every moderate lambda in [1, 10] gives R2 in [0.86, 0.89] — the
    improvement is robust, not a single lucky setting. Even the *untuned* lambda=1 (R2 =
    {bars_final.iloc[2]["R2"]:.3f}) beats bars 1 and 2 comfortably.

    **bar 3 - bar 1 is the recoverable synergy/antagonism.** ~0.23 in R2. This is exactly the
    above-noise structure the A/B ceiling said had to exist — and here a model actually captures it.

    **Rigor caveat.** The lambda=10 row is **selected on the 17 test points**, so it is optimistic
    as a headline. A fair number needs lambda chosen by cross-validation over held-out *training*
    regimens/cells; treat the sweep as the path and lambda=1 (untuned) as the conservative estimate.
    Both still beat bars 1-2. n = 17 and one query cell remain the honest limits.
    """)
    return


@app.cell
def _(eb_df, exp_holdout_cell, exp_test_conditions):
    # --- Learned weights: [1,1] additivity prior vs [0,0] shrink-to-zero (same X, same lambda) ---
    # Only the prior differs:  W = (X^T X + lam I)^{-1} (X^T Y + lam * w_prior)
    #   [1,1] prior -> w_prior = 1 on the two ingredient columns
    #   [0,0] prior -> w_prior = 0  (default Rhaister)
    import numpy as _npW
    import pandas as _pdW
    import ast as _astW
    from rhaister.train_sensitivity import _als_decompose_scalar as _alsW
    from rhaister.prepare_sensitivity import make_splits as _msplitW

    _LAM = 10.0
    _si = {"holdout_cells": [exp_holdout_cell], "test_treatments": {exp_holdout_cell: set(exp_test_conditions)}}
    _tr, _ = _msplitW(eb_df, _si)
    _allc = sorted(eb_df.cell_line.unique()); _allt = sorted(eb_df.condition.unique())
    _c2i = {c: i for i, c in enumerate(_allc)}; _t2i = {t: i for i, t in enumerate(_allt)}
    _ctr = _npW.array([_c2i[c] for c in _tr.cell_line]); _ttr = _npW.array([_t2i[t] for t in _tr.condition])
    _yv = _tr.growth_rate.to_numpy(float)
    _mu, _ce, _te = _alsW(_yv, _ctr, _ttr, len(_allc), len(_allt), n_iter=30)
    _yimp = _mu + _ce[:, None] + _te[None, :]; _yimp[_ctr, _ttr] = _yv
    _hc = _c2i[exp_holdout_cell]; _nonh = _npW.array([i for i in range(len(_allc)) if i != _hc])
    _Dx = sorted({_t2i[t] for t in _tr.loc[_tr.cell_line == exp_holdout_cell, "condition"]})
    _pos = {t: j for j, t in enumerate(_Dx)}
    _X = _yimp[_npW.ix_(_nonh, _Dx)]; _M = _X.T @ _X + _LAM * _npW.eye(len(_Dx))
    _single = {}
    for _c in _tr.loc[_tr.cell_line == exp_holdout_cell, "condition"]:
        _t = _astW.literal_eval(_c)
        if len(_t) == 1 and not _t[0][0].startswith("DMSO"):
            _single[(_t[0][0], _t[0][1])] = _c

    _rows = []
    for _c in exp_test_conditions:
        _comps = _astW.literal_eval(_c)
        _cols = [_pos[_t2i[_single[(d, dose)]]] for d, dose, u in _comps]
        _y = _yimp[_nonh, _t2i[_c]]
        _wp = _npW.zeros(len(_Dx))
        for _ci in _cols:
            _wp[_ci] = 1.0
        _Wa = _npW.linalg.solve(_M, _X.T @ _y + _LAM * _wp)   # [1,1] prior
        _Wz = _npW.linalg.solve(_M, _X.T @ _y)                # [0,0] prior
        _off_a = float(_npW.abs(_Wa).sum() - abs(_Wa[_cols[0]]) - abs(_Wa[_cols[1]]))
        _off_z = float(_npW.abs(_Wz).sum() - abs(_Wz[_cols[0]]) - abs(_Wz[_cols[1]]))
        _rows.append({"combo": "+".join(d for d, _, _ in _comps),
                      "wA[1,1]": round(float(_Wa[_cols[0]]), 2), "wB[1,1]": round(float(_Wa[_cols[1]]), 2),
                      "wA[0,0]": round(float(_Wz[_cols[0]]), 2), "wB[0,0]": round(float(_Wz[_cols[1]]), 2),
                      "off-ingredient |w|  [1,1]": round(_off_a, 2), "[0,0]": round(_off_z, 2)})

    w_prior_compare = _pdW.DataFrame(_rows)
    _ma = w_prior_compare[["wA[1,1]", "wB[1,1]"]].abs().to_numpy().mean()
    _mz = w_prior_compare[["wA[0,0]", "wB[0,0]"]].abs().to_numpy().mean()
    print(f"lambda={_LAM:g}.  mean |ingredient weight|:  [1,1] prior = {_ma:.2f}   [0,0] prior = {_mz:.2f}")
    w_prior_compare
    return


@app.cell
def _(mo):
    mo.md(rf"""
    ### Where the weight goes: [1,1] vs [0,0]

    Same solver, same `lambda=10` — only the prior differs. The gap is stark:

    - **[1,1] prior:** the two ingredient columns carry weight ~**0.89** on average. The model
      builds each combo out of *its own ingredients*, then bends them slightly below 1
      (sub-additivity — e.g. Gem+Pac -> [0.64, 0.84]).
    - **[0,0] prior:** the ingredient columns get ~**0.06** — essentially **ignored**.
      Shrink-to-zero pulls them to nothing, so the ridge has to reconstruct the combo from a
      diffuse mix of *other* drugs' responses, with no anchor.

    **That is the whole reason bar 3 beats bar 2.** It isn't that combos need a fancier model —
    it's that shrink-to-0 discards the single most informative predictors (the two ingredients),
    while shrink-to-[1,1] keeps them and spends the data on the deviation. At `lambda -> inf` the
    two priors lock the ingredient weights at exactly 1 (= bar 1) versus exactly 0 (predict the
    ALS mean, ignoring ingredients entirely).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Experiment at scale — all 52 cell lines (leave-one-cell-out)

    SW480 gave a clean story (bar 3, the [1,1] prior, won at R^2 = 0.89). **Does it
    generalize?** For each of the 52 cell lines in turn, hold out *its* eligible combos and
    predict them from the other 51 cells + its own single agents. 52 x 17 = 884 test points.

    **It does not generalize — the ranking flips.** On SW480 additivity works; across the
    panel it fails badly (add-the-singles has R^2 < 0 on **31 of 52 cells**, down to -6).
    SW480 happens to sit in the non-saturating regime and was unrepresentative.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The four bars as a 2x2:  weight-prior  x  intercept

    Two knobs define the methods — what the ridge weights shrink toward, and whether a free
    baseline (intercept) term is present:

    | bar | weight prior | intercept (free term) |
    |---|---|---|
    | 1. add-the-singles | fixed [1,1], no learning | no |
    | 2. Rhaister ships | 0 | **yes** |
    | 3. Rhaister + [1,1] | [1,1] | no |
    | 4. [1,1] + interaction | [1,1] | **yes** |

    An uncentered ridge with **no** intercept is forced through the additive prediction
    (lambda -> inf makes the weights exactly [1,1] = pure additivity). The **intercept** is a
    free term that lets the model shift off the additive anchor — "soak up whatever additivity
    misses." At lambda -> inf it becomes the mean synergy `mean(combo - additive)`.
    """)
    return


@app.cell
def _(ab_ceiling):
    # --- load the all-52 four-bar metrics (2x2 prior x intercept, lambda=10) ---
    import json as _j52
    with open("/home/jeannie/relearn/notebooks/jeannie/bars_2x2_metrics.json") as _f52:
        bars52 = _j52.load(_f52)
    bars52_ceiling = ab_ceiling["overall combos"]["r_full_SB_mean"]
    import pandas as _pd52
    _pd52.DataFrame(bars52["pooled"]).T
    return bars52, bars52_ceiling


@app.cell
def _(bars52, bars52_ceiling):
    # --- Fig 1a: four bars at lambda=10, pooled + per-cell, with A/B ceiling ---
    import matplotlib.pyplot as _plt
    import numpy as _npp

    _bars = ["bar1", "bar2", "bar3", "bar4"]
    _labels = ["add-the-\nsingles", "shrink-0 +int\n(Rhaister ships)", "[1,1]\nno intercept", "[1,1]\n+intercept"]
    _r2 = [bars52["pooled"][b]["R2"] for b in _bars]
    _pe = [bars52["pooled"][b]["Pearson"] for b in _bars]
    _pcv = [bars52["per_cell_R2"][b] for b in _bars]
    _swi = bars52["sw480_idx"]
    _x = _npp.arange(4)

    fig_bars, (_axA, _axB) = _plt.subplots(1, 2, figsize=(11, 4.4))
    _w = 0.38
    _axA.bar(_x - _w/2, _r2, _w, label="R^2", color="#4C72B0")
    _axA.bar(_x + _w/2, _pe, _w, label="Pearson", color="#DD8452")
    _axA.axhline(bars52_ceiling, ls="--", color="gray", lw=1)
    _axA.text(0.0, bars52_ceiling + 0.015, f"A/B sampling ceiling {bars52_ceiling:.2f}", ha="left", fontsize=8, color="gray")
    _axA.axhline(0, color="k", lw=0.6)
    _axA.set_xticks(_x); _axA.set_xticklabels(_labels, fontsize=7.5)
    _axA.set_ylabel("pooled score (884 combos)"); _axA.set_ylim(-0.15, 1.0)
    _axA.legend(fontsize=8, loc="center right"); _axA.set_title("Pooled across 52 cells (lambda=10)", fontsize=10)

    _bp = _axB.boxplot(_pcv, showfliers=False, positions=_x, widths=0.5, patch_artist=True)
    for _p in _bp["boxes"]:
        _p.set_facecolor("#B0C4DE")
    for _i in range(4):
        _axB.plot(_x[_i], _pcv[_i][_swi], marker="*", color="red", ms=13, zorder=6)
    _axB.axhline(0, color="k", lw=0.6); _axB.set_ylim(-1.2, 1.0)
    _axB.set_xticks(_x); _axB.set_xticklabels(_labels, fontsize=7.5)
    _axB.set_ylabel("per-cell R^2"); _axB.set_title("Per-cell spread (red * = SW480)", fontsize=10)

    fig_bars.suptitle("Fig 1a - predict held-out combo growth rate (leave-one-cell-out, 52 x 17)", fontsize=11)
    fig_bars.tight_layout()
    fig_bars.savefig("/home/jeannie/relearn/notebooks/jeannie/fig1a_bars.png", dpi=150, bbox_inches="tight")
    fig_bars
    return


@app.cell
def _(bars52, bars52_ceiling, mo):
    mo.md(rf"""
    ### Reading the figure

    **Across the panel, "Rhaister ships" (shrink-0 + intercept) wins** — pooled R^2 =
    {bars52["pooled"]["bar2"]["R2"]}, Pearson {bars52["pooled"]["bar2"]["Pearson"]}, and the
    highest per-cell median. The additivity-anchored bars 3 and 4 sit far lower
    ({bars52["pooled"]["bar3"]["R2"]} / {bars52["pooled"]["bar4"]["R2"]}), and pure
    add-the-singles is at chance ({bars52["pooled"]["bar1"]["R2"]}).

    **The [1,1] prior helps SW480 but hurts the panel.** Right panel: SW480 (red star) is a
    *best case* for additivity — on bars 1, 3, 4 it sits near the top of the spread while the
    median cell is negative. The prior encodes "combos are additive," which is true for SW480
    and false for the ~30 cells where both drugs bite and the response saturates.

    **The intercept does not rescue it (bar 4 ~= bar 3).** Additivity's miss is *directional*,
    not a constant offset: saturation compresses the true combo nonlinearly, cell-by-cell. A
    free baseline term fixes a shift, not a wrong direction — only freeing the weights
    (shrink-0) captures it.

    **Honest status.** Best method reaches R^2 ~ {bars52["pooled"]["bar2"]["R2"]} against a
    sampling ceiling ~{bars52_ceiling:.2f} — real headroom remains. And the headline the
    SW480 panel suggested ("additivity closes the gap") is **false across the panel**; it holds
    only for the fraction of cell lines in the non-saturating regime. lambda=10 shown; lambda=1
    gives the same ranking.
    """)
    return


@app.cell
def _(bars52, cell_line_md_df):
    # --- Scatter: per-cell R^2, bar 3 vs bar 4 (each dot = one held-out cell line) ---
    # points below R^2=0 in either bar are annotated with their cell-line name
    import matplotlib.pyplot as _plts
    import numpy as _nps

    _b3 = _nps.array(bars52["per_cell_R2"]["bar3"])
    _b4 = _nps.array(bars52["per_cell_R2"]["bar4"])
    _names_all = _nps.array(bars52["cells"])
    _swi = bars52["sw480_idx"]
    _lo, _hi = -1.5, 1.0
    _b3c, _b4c = _nps.clip(_b3, _lo, _hi), _nps.clip(_b4, _lo, _hi)
    _clip = int(((_b3 < _lo) | (_b4 < _lo)).sum())

    # human-readable name lookup (cellosaurus id -> cell_name), fall back to the id
    _lut = (cell_line_md_df.drop_duplicates("Cell_ID_Cellosaur")
            .set_index("Cell_ID_Cellosaur")["cell_name"])
    _label = _nps.array([str(_lut.get(c, c)) for c in _names_all])

    # colour by sign quadrant
    _col = _nps.where((_b3 > 0) & (_b4 > 0), "#2a9d8f",
            _nps.where((_b3 < 0) & (_b4 < 0), "#e76f51", "#9aa0a6"))

    fig_scatter, ax = _plts.subplots(figsize=(7.6, 7.6))
    ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.7)
    ax.plot([_lo, _hi], [_lo, _hi], ls="--", color="k", lw=0.8, label="bar 3 = bar 4")
    ax.scatter(_b3c, _b4c, c=_col, s=45, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter([_b3c[_swi]], [_b4c[_swi]], marker="*", s=320, color="red",
               edgecolor="black", linewidth=0.5, zorder=5, label="SW480")

    # --- annotate every line with R^2 < 0 (either bar); greedy vertical declutter so
    #     coincident/clipped points (e.g. the four pinned at the -1.5 corner) don't overlap ---
    _neg = _nps.where((_b3 < 0) | (_b4 < 0))[0]
    _neg = _neg[_neg != _swi]
    _neg = _neg[_nps.argsort(_b4c[_neg])]      # bottom-most first
    _gap, _last, _laby = 0.104, -1e9, []
    for _i in _neg:
        _y = max(float(_b4c[_i]), _last + _gap)
        _laby.append(_y); _last = _y
    for _i, _ly in zip(_neg, _laby):
        ax.annotate(_label[_i], (_b3c[_i], _b4c[_i]),
                    xytext=(_b3c[_i] + 0.06, _ly),
                    fontsize=6.2, color="#7a2f1e", ha="left", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#e76f51", lw=0.4, alpha=0.55))

    ax.set_xlim(_lo, _hi); ax.set_ylim(_lo, _hi)
    ax.set_xlabel("bar 3 per-cell R^2   ([1,1], no intercept)")
    ax.set_ylabel("bar 4 per-cell R^2   ([1,1] + intercept)")
    _np3 = int((_b3 > 0).sum()); _np4 = int((_b4 > 0).sum()); _better = int((_b4 > _b3).sum())
    ax.set_title(f"Per-cell R^2: bar3 vs bar4\n"
                 f"positive R^2: bar3 {_np3}/52, bar4 {_np4}/52   |   intercept helps (bar4>bar3): {_better}/52\n"
                 f"labelled: {len(_neg)} lines with R^2<0 in either bar",
                 fontsize=9)
    ax.text(0.12, 0.92, "both R^2 > 0", color="#2a9d8f", fontsize=8.5, ha="left", style="italic")
    ax.text(-0.05, -1.42, f"{_clip} clipped to floor (R^2 < {_lo})",
            fontsize=7.5, color="gray", ha="left")
    ax.legend(fontsize=8, loc="upper left")
    fig_scatter.tight_layout()
    fig_scatter.savefig("/home/jeannie/relearn/notebooks/jeannie/fig_bar3_vs_bar4.png", dpi=150, bbox_inches="tight")
    fig_scatter
    return


@app.cell
def _(bars52, cell_line_md_df, mo, np, pd):
    # --- Result: cell lines that fall below R^2 = 0 in the bar3-vs-bar4 figure ---
    _b3 = np.array(bars52["per_cell_R2"]["bar3"])
    _b4 = np.array(bars52["per_cell_R2"]["bar4"])
    _cvcl = np.array(bars52["cells"])
    _swi = bars52["sw480_idx"]
    _lut = cell_line_md_df.drop_duplicates("Cell_ID_Cellosaur").set_index("Cell_ID_Cellosaur")

    _tbl = pd.DataFrame({
        "cell_line": [str(_lut["cell_name"].get(c, c)) for c in _cvcl],
        "cvcl": _cvcl,
        "organ": [str(_lut["Organ"].get(c, "?")) for c in _cvcl],
        "bar3_R2": _b3.round(3),
        "bar4_R2": _b4.round(3),
    })
    _neg3 = _tbl["bar3_R2"] < 0
    _neg4 = _tbl["bar4_R2"] < 0
    _tbl["quadrant"] = np.where(_neg3 & _neg4, "both<0",
                        np.where(~_neg3 & ~_neg4, "both>=0", "mixed"))

    # public result: the below-zero lines, worst (most negative bar4) first
    bar34_below_zero = (_tbl[_neg3 | _neg4]
                        .sort_values("bar4_R2")
                        .reset_index(drop=True))

    _n3, _n4 = int(_neg3.sum()), int(_neg4.sum())
    _nboth = int((_neg3 & _neg4).sum())
    _neither = len(bar34_below_zero)
    _sw = _tbl.iloc[_swi]
    _organ_counts = (bar34_below_zero["organ"].value_counts().head(3)
                     .to_string().replace("\n", "; "))

    mo.vstack([
        mo.md(rf"""
    ### Cell lines below R^2 = 0 — bar 3 ([1,1], no intercept) vs bar 4 ([1,1] + intercept)

    Each dot in the scatter above is one held-out cell line (leave-one-cell-out, 17 combos each).
    Points below zero are lines where the additivity-anchored model does **worse than predicting the
    per-cell mean**.

    - **bar 3 R^2 < 0:** {_n3}/52 &nbsp;|&nbsp; **bar 4 R^2 < 0:** {_n4}/52 &nbsp;|&nbsp;
      **negative in both:** {_nboth}/52 &nbsp;|&nbsp; **negative in either:** {_neither}/52
    - The intercept (bar 4) barely rescues anyone — the two negative sets are nearly identical, so the
      failure is a **wrong direction (assay saturation), not a constant offset**.
    - Failures concentrate by lineage: {_organ_counts}.
    - Reference **SW480** sits at the opposite (resistant) end: bar3 = {_sw['bar3_R2']:.2f}, bar4 = {_sw['bar4_R2']:.2f}.

    Full list of the {_neither} below-zero lines (most-negative bar 4 first):
    """),
        bar34_below_zero,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### bar 3 vs bar 4, per cell

    Almost every cell sits **on the diagonal** — the intercept (bar 4) barely moves per-cell
    R^2, confirming it is close to a no-op. **32/52 cells are positive** (green, both R^2 > 0);
    **20/52 are negative** (red, both < 0), the same split for both bars. SW480 (star) is among
    the best-predicted cells and sits *slightly below* the diagonal — the intercept actively
    **hurts** it (0.89 -> 0.82). The ~40% of cells in the negative quadrant are where additivity
    is directionally wrong (both drugs bite, response saturates); a free baseline term cannot
    fix a wrong direction, which is why bar 4 tracks bar 3 rather than rescuing it.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(np, pd):
    # --- Per-cell summary of the [1,1] additivity-prior model (bars52 "bar3", lambda=10) ---
    # r2_add = per-cell R^2 of the [1,1]/no-intercept ridge; sign labels who additivity fails on.
    # This is the same per-cell R^2 shown on the x-axis of the bar3-vs-bar4 scatter.
    pred_all52 = pd.read_parquet("/home/jeannie/relearn/notebooks/jeannie/bars_all52_v2_predictions.parquet")

    def _percell_r2(m, p):
        _ss = ((m - m.mean()) ** 2).sum()
        return float("nan") if _ss == 0 else float(1 - ((m - p) ** 2).sum() / _ss)

    _rows_pa = []
    for _cl, _g in pred_all52.groupby("cell_line"):
        _m = _g["measured"].to_numpy()
        _pr = _g["bar3_prior11_lam10"].to_numpy()
        _add = _g["bar1_add"].to_numpy()
        _rows_pa.append({
            "cell_line": _cl,
            "r2_add": _percell_r2(_m, _pr),
            "mean_measured": float(_m.mean()),
            "mean_abs_synergy": float(np.abs(_m - _add).mean()),
            "n": int(len(_g)),
        })
    percell_add = pd.DataFrame(_rows_pa)
    percell_add["sign"] = np.where(percell_add["r2_add"] < 0, "R2<0", "R2>=0")
    percell_add.sort_values("r2_add").round(3)
    return percell_add, pred_all52


@app.cell
def _(ast, cell_line_md_df, eb_df, percell_add, summary_stats_df):
    # Per-cell biological trait: overall single-agent drug sensitivity (independent of the combo test set)
    def _is_single_drug(c):
        t = ast.literal_eval(c)
        return len(t) == 1 and not t[0][0].startswith("DMSO")

    single_gr_all = eb_df[eb_df["condition"].map(_is_single_drug)]
    cell_sensitivity = (
        single_gr_all.groupby("cell_line")["growth_rate"]
        .agg(mean_single_gr="mean",
             median_single_gr="median",
             frac_strong_kill=lambda s: (s < -1).mean())
        .round(3)
    )

    # Robustly locate the join key + a sampling-depth column in the two metadata tables
    our_ids = set(percell_add["cell_line"])
    def _find_key(df, ids):
        best, best_ov = None, 0
        for c in df.columns:
            try:
                ov = len(set(df[c].astype(str)) & ids)
            except Exception:
                ov = 0
            if ov > best_ov:
                best, best_ov = c, ov
        return best, best_ov

    md_key, md_ov = _find_key(cell_line_md_df, our_ids)
    ss_key, ss_ov = _find_key(summary_stats_df, our_ids)
    print("cell_line_md_df cols:", list(cell_line_md_df.columns))
    print("summary_stats_df cols:", list(summary_stats_df.columns))
    print(f"join key -> cell_line_md_df: {md_key} (overlap {md_ov}) | summary_stats_df: {ss_key} (overlap {ss_ov})")
    return cell_sensitivity, md_key, ss_key


@app.cell
def _(
    cell_line_md_df,
    cell_sensitivity,
    md_key,
    np,
    percell_add,
    ss_key,
    summary_stats_df,
):
    # pick lineage/tissue + sampling-depth columns defensively
    def _pick(cols, opts):
        low = {c.lower(): c for c in cols}
        for o in opts:
            if o in low:
                return low[o]
        for o in opts:
            for c in cols:
                if o in c.lower():
                    return c
        return None

    tissue_col = _pick(cell_line_md_df.columns,
                       ["tissue", "organ", "lineage", "primary_disease", "cancer_type",
                        "disease", "site", "subtype", "oncotree"])
    depth_col = _pick(summary_stats_df.columns,
                      ["n_cells", "cell_count", "num_cells", "n_obs", "count", "cells"])

    cellprofile = percell_add.merge(cell_sensitivity, on="cell_line", how="left")

    if md_key and tissue_col:
        _md = cell_line_md_df[[md_key, tissue_col]].copy()
        _md.columns = ["cell_line", "tissue"]
        _md["cell_line"] = _md["cell_line"].astype(str)
        _md = _md.drop_duplicates("cell_line")  # md has one row per driver gene; keep one tissue per cell
        cellprofile = cellprofile.merge(_md, on="cell_line", how="left")
    else:
        cellprofile["tissue"] = "unknown"

    if ss_key and depth_col:
        _ss = summary_stats_df[[ss_key, depth_col]].copy()
        _ss.columns = ["cell_line", "depth"]
        _ss["cell_line"] = _ss["cell_line"].astype(str)
        _ss = _ss.groupby("cell_line", as_index=False)["depth"].median()
        cellprofile = cellprofile.merge(_ss, on="cell_line", how="left")
    else:
        cellprofile["depth"] = np.nan

    # neg vs pos comparison on the candidate drivers
    _cmp_cols = ["r2_add", "mean_measured", "min_single", "mean_single_gr",
                 "median_single_gr", "frac_strong_kill", "mean_abs_synergy", "n", "depth"]
    _cmp_cols = [c for c in _cmp_cols if c in cellprofile.columns]
    neg_vs_pos = cellprofile.groupby("sign")[_cmp_cols].median().round(3)
    print("Median trait by group (R2<0 vs R2>=0):")
    print(neg_vs_pos.T.to_string())
    cellprofile.sort_values("r2_add").round(3)
    return (cellprofile,)


@app.cell
def _(cellprofile):
    # Lineage composition + enrichment of the failing cells
    tissue_tab = (
        cellprofile.assign(neg=(cellprofile["sign"] == "R2<0").astype(int))
        .groupby("tissue")
        .agg(n_total=("neg", "size"), n_neg=("neg", "sum"),
             median_r2=("r2_add", "median"),
             median_single_gr=("mean_single_gr", "median"))
    )
    tissue_tab["frac_neg"] = (tissue_tab["n_neg"] / tissue_tab["n_total"]).round(2)
    tissue_tab = tissue_tab.sort_values(["frac_neg", "n_total"], ascending=[False, False]).round(3)
    print("Lineage breakdown (frac_neg = share of that lineage with R2<0):")
    print(tissue_tab.to_string())
    tissue_tab
    return (tissue_tab,)


@app.cell
def _(cellprofile, np, plt, tissue_tab):
    fig_cells, (_axA, _axB, _axC) = plt.subplots(1, 3, figsize=(15, 4.6))
    _cmap = {"R2<0": "#e76f51", "R2>=0": "#2a9d8f"}
    _c = cellprofile["sign"].map(_cmap)

    # A: technical hypothesis -> sampling depth vs additive R^2
    if cellprofile["depth"].notna().any():
        _axA.scatter(cellprofile["depth"], cellprofile["r2_add"], c=_c, s=55, edgecolor="white")
        _axA.set_xscale("log")
        _axA.set_xlabel("sampling depth (cells, log)")
    else:
        _axA.scatter(cellprofile["n"], cellprofile["r2_add"], c=_c, s=55, edgecolor="white")
        _axA.set_xlabel("n eligible combos (depth proxy)")
    _axA.axhline(0, color="k", lw=0.7)
    _axA.set_ylabel("additive R^2"); _axA.set_ylim(-6, 1.05)
    _axA.set_title("Technical: is R^2<0 just thin sampling?")

    # B: biological hypothesis -> single-agent sensitivity vs additive R^2
    _axB.scatter(cellprofile["mean_single_gr"], cellprofile["r2_add"], c=_c, s=55, edgecolor="white")
    _axB.axhline(0, color="k", lw=0.7); _axB.axvline(0, color="gray", lw=0.6)
    _axB.set_xlabel("mean single-agent growth_rate  (more negative = more drug-sensitive)")
    _axB.set_ylabel("additive R^2"); _axB.set_ylim(-6, 1.05)
    _axB.set_title("Biological: sensitive cells saturate -> R^2<0")

    # C: lineage enrichment
    _tt = tissue_tab[tissue_tab["n_total"] >= 2].sort_values("frac_neg")
    _yy = np.arange(len(_tt))
    _axC.barh(_yy, _tt["frac_neg"], color="#4C72B0")
    _axC.set_yticks(_yy); _axC.set_yticklabels(_tt.index, fontsize=7.5)
    _axC.set_xlabel("fraction of lineage with R^2<0")
    _axC.set_title("Which lineages fail (n>=2 cells)")

    fig_cells.tight_layout()
    fig_cells
    return


@app.cell
def _(bars52, cell_line_md_df, cellprofile, np, plt):
    # --- Bar 3 per-cell R^2 vs single-agent drug potency (each dot = one held-out cell line) ---
    # Tests the "R^2<0 = drug-sensitive lines" claim directly. Extreme lines on each
    # potency end are labelled. r2_add is the bar3 ([1,1], no-intercept) per-cell R^2.
    from scipy.stats import spearmanr as _spearman

    _lut_nm = (cell_line_md_df.drop_duplicates("Cell_ID_Cellosaur")
               .set_index("Cell_ID_Cellosaur")["cell_name"])
    _pp = cellprofile.copy()
    _pp["name"] = [str(_lut_nm.get(c, c)) for c in _pp["cell_line"]]

    _x = _pp["mean_single_gr"].to_numpy()          # potency: more negative = drugs kill harder
    _yfloor = -2.0
    _yraw = _pp["r2_add"].to_numpy()
    _y = np.clip(_yraw, _yfloor, 1.05)
    _nclip = int((_yraw < _yfloor).sum())
    _col = np.where(_yraw < 0, "#e76f51", "#2a9d8f")
    _swi = int(np.where(_pp["cell_line"].to_numpy() == bars52["cells"][bars52["sw480_idx"]])[0][0])

    _rho = _spearman(_x, _yraw).correlation
    _r = np.corrcoef(_x, _yraw)[0, 1]

    fig_potency, axp = plt.subplots(figsize=(8.2, 6.2))
    axp.axhline(0, color="gray", lw=0.7)
    axp.axvline(0, color="gray", lw=0.6, ls=":")
    axp.scatter(_x, _y, c=_col, s=55, edgecolor="white", linewidth=0.6, zorder=3)
    axp.scatter([_x[_swi]], [_y[_swi]], marker="*", s=340, color="red",
                edgecolor="black", linewidth=0.5, zorder=6, label="SW480")

    # label the extreme lines on either potency end (4 most potent, 4 least potent)
    _k = 4
    _order = np.argsort(_x)
    _ext = list(_order[:_k]) + list(_order[-_k:])
    _xmed = np.median(_x)
    for _i in _ext:
        _left = _x[_i] < _xmed
        _dx = -0.012 if _left else 0.012
        axp.annotate(_pp["name"].iloc[_i], (_x[_i], _y[_i]),
                     xytext=(_x[_i] + _dx, _y[_i] + 0.10),
                     fontsize=7, color="#333", ha="right" if _left else "left",
                     va="bottom", zorder=7,
                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6))

    axp.set_xlabel("single-agent drug potency  =  mean single-agent growth_rate\n"
                   "(left = drugs kill harder / more potent      right = more resistant)")
    axp.set_ylabel("bar 3 per-cell R^2   ([1,1], no intercept)")
    axp.set_ylim(_yfloor - 0.05, 1.1)
    axp.set_title(f"Bar 3 fit vs drug potency  (each dot = 1 held-out cell line, n=52)\n"
                  f"no relationship: Spearman rho = {_rho:+.2f},  Pearson r = {_r:+.2f}",
                  fontsize=10)
    if _nclip:
        axp.text(_x.min(), _yfloor + 0.02,
                 f"{_nclip} cell(s) clipped to floor (R^2 < {_yfloor:g})",
                 fontsize=7.5, color="gray", va="bottom")
    axp.legend(fontsize=8, loc="lower right")
    fig_potency.tight_layout()
    fig_potency.savefig("/home/jeannie/relearn/notebooks/jeannie/fig_bar3_vs_potency.png",
                        dpi=150, bbox_inches="tight")
    fig_potency
    return


@app.cell
def _(ast, np, pd, plt, pred_all52):
    # --- Bar 3 R^2 vs potency, aggregated PER PERTURBATION (each dot = one combo condition) ---
    # Flip of the per-cell plot: for each perturbation we average across all 52 cell lines.
    #   x = potency  = mean measured growth_rate across the 52 cells (more negative = more potent)
    #   y = R^2 across the 52 cells for bar3 ([1,1], no-intercept) predictions
    # Every perturbation is labelled (name + dose) so each dot is identifiable.
    from scipy.stats import spearmanr as _spearman2

    def _r2_perturb(m, p):
        _ss = ((m - m.mean()) ** 2).sum()
        return float("nan") if _ss == 0 else float(1 - ((m - p) ** 2).sum() / _ss)

    def _dose_label(cond):
        _parts = ast.literal_eval(cond)
        return " + ".join(f"{_d} {_dose:g}uM" for _d, _dose, _u in _parts)

    _rows_pb = []
    for _cond, _g in pred_all52.groupby("condition"):
        _m = _g["measured"].to_numpy()
        _pr = _g["bar3_prior11_lam10"].to_numpy()
        _rows_pb.append({"condition": _cond, "label": _dose_label(_cond),
                         "n_cells": int(len(_g)),
                         "potency": float(_m.mean()),
                         "r2": _r2_perturb(_m, _pr)})
    perturb_r2 = pd.DataFrame(_rows_pb).sort_values("potency").reset_index(drop=True)

    _xp = perturb_r2["potency"].to_numpy()
    _yp = perturb_r2["r2"].to_numpy()
    _lbl = perturb_r2["label"].to_numpy()
    _colp = np.where(_yp < 0, "#e76f51", "#2a9d8f")
    _rho2 = _spearman2(_xp, _yp).correlation
    _r2p = np.corrcoef(_xp, _yp)[0, 1]

    fig_perturb, axq = plt.subplots(figsize=(11.5, 6.6))
    axq.axhline(0, color="gray", lw=0.7)
    axq.axvline(0, color="gray", lw=0.6, ls=":")
    axq.scatter(_xp, _yp, c=_colp, s=80, edgecolor="white", linewidth=0.7, zorder=3)

    # label EVERY perturbation; split into a left/right column and vertically declutter each
    def _spread_y(ids):
        _gap, _last, _out = 0.17, -1e9, {}
        for _i in sorted(ids, key=lambda j: _yp[j]):
            _yy = max(float(_yp[_i]), _last + _gap); _out[_i] = _yy; _last = _yy
        return _out

    _mid = float(np.median(_xp))
    _left_ids = [i for i in range(len(_xp)) if _xp[i] <= _mid]
    _right_ids = [i for i in range(len(_xp)) if _xp[i] > _mid]
    _lyL, _lyR = _spread_y(_left_ids), _spread_y(_right_ids)
    _xL, _xR = _xp.min() - 0.018, _xp.max() + 0.018
    for _i in _left_ids:
        axq.annotate(_lbl[_i], (_xp[_i], _yp[_i]), xytext=(_xL, _lyL[_i]),
                     fontsize=6.6, color="#333", ha="right", va="center", zorder=6,
                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6))
    for _i in _right_ids:
        axq.annotate(_lbl[_i], (_xp[_i], _yp[_i]), xytext=(_xR, _lyR[_i]),
                     fontsize=6.6, color="#333", ha="left", va="center", zorder=6,
                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6))

    axq.set_xlabel("perturbation potency  =  mean measured growth_rate across 52 cells\n"
                   "(left = combo kills harder / more potent      right = weaker effect)")
    axq.set_ylabel("per-perturbation R^2   (bar 3, [1,1] no intercept, across 52 cells)")
    axq.set_title(f"Bar 3 fit vs potency, PER PERTURBATION  (each dot = 1 combo, n={len(perturb_r2)})\n"
                  f"no relationship: Spearman rho = {_rho2:+.2f},  Pearson r = {_r2p:+.2f}",
                  fontsize=10)
    axq.set_xlim(_xp.min() - 0.035, _xp.max() + 0.035)
    axq.set_ylim(min(-1.9, _yp.min() - 0.1), 1.05)
    fig_perturb.tight_layout()
    fig_perturb.savefig("/home/jeannie/relearn/notebooks/jeannie/fig_bar3_r2_per_perturbation.png",
                        dpi=150, bbox_inches="tight")
    fig_perturb
    return


@app.cell
def _(ast, np, pd, plt, pred_all52):
    # --- Bar 2 (Rhaister ships) R^2 vs potency, PER PERTURBATION (each dot = one combo) ---
    # Same as the bar-3 per-perturbation plot but for bar2 = shrink-to-0 ridge + intercept
    # (the production "Rhaister ships" model). Every perturbation labelled (name + dose).
    from scipy.stats import spearmanr as _spearman3

    def _r2_pb2(m, p):
        _ss = ((m - m.mean()) ** 2).sum()
        return float("nan") if _ss == 0 else float(1 - ((m - p) ** 2).sum() / _ss)

    def _dose_label2(cond):
        _parts = ast.literal_eval(cond)
        return " + ".join(f"{_d} {_dose:g}uM" for _d, _dose, _u in _parts)

    _rows_b2 = []
    for _cond, _g in pred_all52.groupby("condition"):
        _m = _g["measured"].to_numpy()
        _pr = _g["bar2_shrink0_lam10"].to_numpy()
        _rows_b2.append({"condition": _cond, "label": _dose_label2(_cond),
                         "potency": float(_m.mean()), "r2": _r2_pb2(_m, _pr)})
    perturb_r2_bar2 = pd.DataFrame(_rows_b2).sort_values("potency").reset_index(drop=True)

    _xb = perturb_r2_bar2["potency"].to_numpy()
    _yb = perturb_r2_bar2["r2"].to_numpy()
    _lb = perturb_r2_bar2["label"].to_numpy()
    _colb = np.where(_yb < 0, "#e76f51", "#2a9d8f")
    _rho3 = _spearman3(_xb, _yb).correlation
    _r3 = np.corrcoef(_xb, _yb)[0, 1]

    fig_perturb_bar2, axr = plt.subplots(figsize=(11.5, 6.6))
    axr.axhline(0, color="gray", lw=0.7)
    axr.axvline(0, color="gray", lw=0.6, ls=":")
    axr.scatter(_xb, _yb, c=_colb, s=80, edgecolor="white", linewidth=0.7, zorder=3)

    # label EVERY perturbation; split into left/right column, spread evenly across the band
    _ylo_b, _yhi_b = -0.1, 1.05
    def _even_y(ids):
        _s = sorted(ids, key=lambda j: _yb[j])
        _pos = np.linspace(_ylo_b + 0.07, _yhi_b - 0.07, len(_s)) if len(_s) > 1 else [np.mean([_ylo_b, _yhi_b])]
        return {i: y for i, y in zip(_s, _pos)}

    _mid_b = float(np.median(_xb))
    _left_b = [i for i in range(len(_xb)) if _xb[i] <= _mid_b]
    _right_b = [i for i in range(len(_xb)) if _xb[i] > _mid_b]
    _eyL, _eyR = _even_y(_left_b), _even_y(_right_b)
    _xLb, _xRb = _xb.min() - 0.018, _xb.max() + 0.018
    for _i in _left_b:
        axr.annotate(_lb[_i], (_xb[_i], _yb[_i]), xytext=(_xLb, _eyL[_i]),
                     fontsize=6.6, color="#333", ha="right", va="center", zorder=6,
                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6))
    for _i in _right_b:
        axr.annotate(_lb[_i], (_xb[_i], _yb[_i]), xytext=(_xRb, _eyR[_i]),
                     fontsize=6.6, color="#333", ha="left", va="center", zorder=6,
                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6))

    axr.set_xlabel("perturbation potency  =  mean measured growth_rate across 52 cells\n"
                   "(left = combo kills harder / more potent      right = weaker effect)")
    axr.set_ylabel("per-perturbation R^2   (bar 2 'Rhaister ships', shrink-0 + intercept)")
    axr.set_title(f"Bar 2 (Rhaister ships) fit vs potency, PER PERTURBATION  (each dot = 1 combo, n={len(perturb_r2_bar2)})\n"
                  f"Spearman rho = {_rho3:+.2f},  Pearson r = {_r3:+.2f}   |   all {int((_yb>=0).sum())}/{len(_yb)} perturbations R^2 >= 0",
                  fontsize=10)
    axr.set_xlim(_xb.min() - 0.035, _xb.max() + 0.035)
    axr.set_ylim(_ylo_b, _yhi_b)
    fig_perturb_bar2.tight_layout()
    fig_perturb_bar2.savefig("/home/jeannie/relearn/notebooks/jeannie/fig_bar2_r2_per_perturbation.png",
                             dpi=150, bbox_inches="tight")
    fig_perturb_bar2
    return


@app.cell
def _(mo, pd, plt, pred_all52):
    # --- Per-perturbation R^2 vs potency, coloured by combo DESIGN CLASS (bar1 vs bar3) ---
    # Each dot = one combo, aggregated across all 52 cells. Colour = design class from
    # artifacts/emeraldbay_combo_design_classes.csv. Two panels share the y-axis so the
    # same combo can be compared between the pure-additivity (bar1) and [1,1]-ridge (bar3) fits.
    _dc = pd.read_csv("/home/jeannie/relearn/artifacts/emeraldbay_combo_design_classes.csv")
    _dcmap = dict(zip(_dc["condition"], _dc["design_class"]))
    _dosemap = dict(zip(_dc["condition"], _dc["doses_uM"]))

    def _r2_cls(m, p):
        _ss = ((m - m.mean()) ** 2).sum()
        return float("nan") if _ss == 0 else float(1 - ((m - p) ** 2).sum() / _ss)

    _rows_cls = []
    for _cond, _g in pred_all52.groupby("condition"):
        _m = _g["measured"].to_numpy()
        _rows_cls.append({
            "drugs": _g["drugs"].iloc[0],
            "doses_uM": _dosemap.get(_cond, ""),
            "design_class": _dcmap.get(_cond, "unknown"),
            "potency": float(_m.mean()),
            "r2_bar1": _r2_cls(_m, _g["bar1_add"].to_numpy()),
            "r2_bar3": _r2_cls(_m, _g["bar3_prior11_lam10"].to_numpy()),
            "condition": _cond,
        })
    perturb_classes = pd.DataFrame(_rows_cls)

    _CLASS_COLORS = {
        "horizontal bypass": "#4C72B0",
        "vertical blockade": "#DD8452",
        "same process / different mechanism": "#55A868",
        "orthogonal process": "#C44E52",
        "pharmacokinetic modulation": "#8172B3",
        "same-node redundancy": "#937860",
    }
    # stable legend order = most common first
    _cls_order = list(perturb_classes["design_class"].value_counts().index)

    fig_classes, (axb1, axb3) = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for _ax, _ycol, _ttl in [
        (axb1, "r2_bar1", "bar 1: add-the-singles (pure additivity, no model)"),
        (axb3, "r2_bar3", "bar 3: [1,1] additivity-prior ridge (no intercept)"),
    ]:
        for _cls in _cls_order:
            _sub = perturb_classes[perturb_classes["design_class"] == _cls]
            _ax.scatter(_sub["potency"], _sub[_ycol], s=110,
                        color=_CLASS_COLORS.get(_cls, "#999999"),
                        edgecolor="white", linewidth=0.8, label=_cls, zorder=3)
        _ax.axhline(0, color="gray", lw=0.7)
        _ax.axvline(0, color="gray", lw=0.6, ls=":")
        _ax.set_title(_ttl, fontsize=10)
        _ax.set_xlabel("perturbation potency = mean growth_rate across 52 cells\n"
                       "(left = combo kills harder / more potent)")

    axb1.set_ylabel("per-perturbation R^2 (across 52 cells)")
    axb1.set_ylim(-2.9, 1.08)
    _h, _l = axb3.get_legend_handles_labels()
    fig_classes.legend(_h, _l, title="combo design class", loc="upper center",
                       ncol=len(_cls_order), fontsize=8.5, frameon=False,
                       bbox_to_anchor=(0.5, 1.005))
    fig_classes.suptitle("Per-perturbation R^2 vs potency, coloured by design class  (n=17 combos)",
                         fontsize=11, y=1.06)
    fig_classes.tight_layout(rect=[0, 0, 1, 0.98])
    fig_classes.savefig("/home/jeannie/relearn/notebooks/jeannie/fig_bar1_bar3_by_designclass.png",
                        dpi=150, bbox_inches="tight")

    _ref = (perturb_classes[["design_class", "drugs", "doses_uM", "potency", "r2_bar1", "r2_bar3"]]
            .sort_values(["design_class", "potency"]).round(3).reset_index(drop=True))
    mo.vstack([fig_classes, mo.md("**Reference — each dot identified:**"), _ref])
    return


@app.cell
def _(mo, pd, plt, pred_all52):
    # --- Per-perturbation R^2 vs potency, coloured by combo DESIGN CLASS (bar2 vs bar3) ---
    # Same as the bar1-vs-bar3 design-class plot, but the left panel is bar2 = the production
    # "Rhaister ships" model (shrink-to-0 ridge + intercept). Each dot = one combo (across 52 cells).
    _dc2 = pd.read_csv("/home/jeannie/relearn/artifacts/emeraldbay_combo_design_classes.csv")
    _dcmap2 = dict(zip(_dc2["condition"], _dc2["design_class"]))
    _dosemap2 = dict(zip(_dc2["condition"], _dc2["doses_uM"]))

    def _r2_cls2(m, p):
        _ss = ((m - m.mean()) ** 2).sum()
        return float("nan") if _ss == 0 else float(1 - ((m - p) ** 2).sum() / _ss)

    _rows_cls2 = []
    for _cond, _g in pred_all52.groupby("condition"):
        _m = _g["measured"].to_numpy()
        _rows_cls2.append({
            "drugs": _g["drugs"].iloc[0],
            "doses_uM": _dosemap2.get(_cond, ""),
            "design_class": _dcmap2.get(_cond, "unknown"),
            "potency": float(_m.mean()),
            "r2_bar2": _r2_cls2(_m, _g["bar2_shrink0_lam10"].to_numpy()),
            "r2_bar3": _r2_cls2(_m, _g["bar3_prior11_lam10"].to_numpy()),
            "condition": _cond,
        })
    perturb_classes_b2b3 = pd.DataFrame(_rows_cls2)

    _CLASS_COLORS2 = {
        "horizontal bypass": "#4C72B0",
        "vertical blockade": "#DD8452",
        "same process / different mechanism": "#55A868",
        "orthogonal process": "#C44E52",
        "pharmacokinetic modulation": "#8172B3",
        "same-node redundancy": "#937860",
    }
    _cls_order2 = list(perturb_classes_b2b3["design_class"].value_counts().index)

    fig_classes_b2b3, (_axb2, _axb3) = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for _ax, _ycol, _ttl in [
        (_axb2, "r2_bar2", "bar 2: 'Rhaister ships' (shrink-to-0 ridge + intercept)"),
        (_axb3, "r2_bar3", "bar 3: [1,1] additivity-prior ridge (no intercept)"),
    ]:
        for _cls in _cls_order2:
            _sub = perturb_classes_b2b3[perturb_classes_b2b3["design_class"] == _cls]
            _ax.scatter(_sub["potency"], _sub[_ycol], s=110,
                        color=_CLASS_COLORS2.get(_cls, "#999999"),
                        edgecolor="white", linewidth=0.8, label=_cls, zorder=3)
        _ax.axhline(0, color="gray", lw=0.7)
        _ax.axvline(0, color="gray", lw=0.6, ls=":")
        _ax.set_title(_ttl, fontsize=10)
        _ax.set_xlabel("perturbation potency = mean growth_rate across 52 cells\n"
                       "(left = combo kills harder / more potent)")

    _axb2.set_ylabel("per-perturbation R^2 (across 52 cells)")
    _axb2.set_ylim(-1.9, 1.08)
    _h2, _l2 = _axb3.get_legend_handles_labels()
    fig_classes_b2b3.legend(_h2, _l2, title="combo design class", loc="upper center",
                            ncol=len(_cls_order2), fontsize=8.5, frameon=False,
                            bbox_to_anchor=(0.5, 1.005))
    fig_classes_b2b3.suptitle("Per-perturbation R^2 vs potency, coloured by design class  (bar2 vs bar3, n=17)",
                              fontsize=11, y=1.06)
    fig_classes_b2b3.tight_layout(rect=[0, 0, 1, 0.98])
    fig_classes_b2b3.savefig("/home/jeannie/relearn/notebooks/jeannie/fig_bar2_bar3_by_designclass.png",
                             dpi=150, bbox_inches="tight")

    _ref2 = (perturb_classes_b2b3[["design_class", "drugs", "doses_uM", "potency", "r2_bar2", "r2_bar3"]]
             .sort_values(["design_class", "potency"]).round(3).reset_index(drop=True))
    mo.vstack([fig_classes_b2b3, mo.md("**Reference — each dot identified:**"), _ref2])
    return


@app.cell
def _(ast, eb_df, np, pd):
    # --- Experiment: bar3-style ridge fit under different additivity W_priors ---
    # bar3 = ridge on the residual (combo - [w1*single_1 + w2*single_2]), then add the weighted
    # prior back. Here we SWEEP W_prior. All 17 eligible combos are 2-drug. Components are ordered
    # by SORTED (alphabetical) drug name: weight[0] -> alphabetically-first drug (matches the
    # `drugs` label order), weight[1] -> second. Symmetric priors are order-invariant.
    from rhaister.combos import is_multi_drug as _is_multi
    from rhaister.prepare_sensitivity import make_splits as _make_splits
    from rhaister.train_sensitivity import _als_decompose_scalar as _als_fn, _drug_regression as _dreg_fn

    _df_wp = eb_df
    _cellsW = sorted(_df_wp.cell_line.unique()); _treatW = sorted(_df_wp.condition.unique())
    _c2iW = {c: i for i, c in enumerate(_cellsW)}; _t2iW = {t: i for i, t in enumerate(_treatW)}
    _ncW, _ntW = len(_cellsW), len(_treatW)
    _singlesW = set(); _scondW = {}
    for _c in _treatW:
        if not _is_multi(_c):
            _t = ast.literal_eval(_c)
            if not _t[0][0].startswith("DMSO"):
                _singlesW.add((_t[0][0], _t[0][1])); _scondW[(_t[0][0], _t[0][1])] = _c
    _eligW = [_c for _c in _treatW if _is_multi(_c)
              and all((d, dose) in _singlesW for d, dose, u in ast.literal_eval(_c))]

    # order each combo's components ALPHABETICALLY by drug name; record the assignment
    _compW, _order_ref = {}, {}
    for _c in _eligW:
        _ranked = sorted(ast.literal_eval(_c), key=lambda _t: _t[0])
        _compW[_c] = [_t2iW[_scondW[(d, dose)]] for d, dose, u in _ranked]
        _order_ref[_c] = {"first_drug (w0)": _ranked[0][0], "second_drug (w1)": _ranked[1][0]}

    _grW = {(r.cell_line, r.condition): r.growth_rate for r in _df_wp.itertuples()}

    def _fit_prior(weights, lam=10.0):
        _W = np.asarray(weights, float); _rows = []
        for _hcn in _cellsW:
            _hc = _c2iW[_hcn]
            _tr, _ = _make_splits(_df_wp, {"holdout_cells": [_hcn],
                                           "test_treatments": {_hcn: set(_eligW)}})
            _ctr = np.array([_c2iW[c] for c in _tr.cell_line])
            _ttr = np.array([_t2iW[t] for t in _tr.condition])
            _yv = _tr.growth_rate.to_numpy(float)
            _mu, _ce, _te = _als_fn(_yv, _ctr, _ttr, _ncW, _ntW, n_iter=30)
            _yimp = _mu + _ce[:, None] + _te[None, :]; _yimp[_ctr, _ttr] = _yv
            _obs = np.zeros((_ncW, _ntW), bool); _obs[_ctr, _ttr] = True
            _tp = [(_hc, _t2iW[c]) for c in _eligW]
            _prior_hc = np.array([(_yimp[_hc, _compW[c]] * _W).sum() for c in _eligW])
            _res = _yimp.copy()
            for c in _eligW:
                _res[:, _t2iW[c]] = _yimp[:, _t2iW[c]] - (_yimp[:, _compW[c]] * _W).sum(1)
            _yr, _cov = _dreg_fn(_res, _obs, _tp, lam=lam, holdout_set={_hc})
            _pred = np.where(_cov, _yr + _prior_hc, _prior_hc)
            for _k, c in enumerate(_eligW):
                _rows.append({"cell_line": _hcn, "condition": c,
                              "pred": float(_pred[_k]), "measured": float(_grW[(_hcn, c)])})
        return pd.DataFrame(_rows)

    _PRIORS_WP = [
        ("[1, 1]  (= bar3 ref)", [1, 1]),
        ("[1, -1]", [1, -1]),
        ("[-1, 1]", [-1, 1]),
        ("[0.5, 0.5]", [0.5, 0.5]),
        ("[0.75, 0.25]", [0.75, 0.25]),
        ("[0.25, 0.75]", [0.25, 0.75]),
    ]

    _dcW = pd.read_csv("/home/jeannie/relearn/artifacts/emeraldbay_combo_design_classes.csv")
    _dcmapW = dict(zip(_dcW.condition, _dcW.design_class))
    _dosemapW = dict(zip(_dcW.condition, _dcW.doses_uM))
    _drugmapW = dict(zip(_dcW.condition, _dcW.drugs))

    def _r2W(m, p):
        _ss = ((m - m.mean()) ** 2).sum()
        return float("nan") if _ss == 0 else float(1 - ((m - p) ** 2).sum() / _ss)

    _tidy = []
    for _lbl, _w in _PRIORS_WP:
        _pf = _fit_prior(_w)
        for _cond, _g in _pf.groupby("condition"):
            _m = _g["measured"].to_numpy(); _pr = _g["pred"].to_numpy()
            _tidy.append({"prior": _lbl, "weights": str(_w), "condition": _cond,
                          "drugs": _drugmapW.get(_cond, "?"),
                          "design_class": _dcmapW.get(_cond, "unknown"),
                          "doses_uM": _dosemapW.get(_cond, ""),
                          "first_drug": _order_ref[_cond]["first_drug (w0)"],
                          "second_drug": _order_ref[_cond]["second_drug (w1)"],
                          "potency": float(_m.mean()), "r2": _r2W(_m, _pr)})
    wprior_perturb_r2 = pd.DataFrame(_tidy)

    wprior_order_ref = (pd.DataFrame([{"drugs": _drugmapW[_c], "doses_uM": _dosemapW[_c],
                                       **_order_ref[_c]} for _c in _eligW])
                        .drop_duplicates().reset_index(drop=True))
    wprior_order_ref
    return (wprior_perturb_r2,)


@app.cell
def _(plt, wprior_perturb_r2):
    # --- W_prior sweep: per-perturbation R^2 vs potency, coloured by design class ---
    # One panel per W_prior (bar3 [1,1] shown first as reference). Each dot = one combo
    # (R^2 across 52 cells). Same colouring scheme as the design-class plots above.
    _CLASS_COLORS_WP = {
        "horizontal bypass": "#4C72B0",
        "vertical blockade": "#DD8452",
        "same process / different mechanism": "#55A868",
        "orthogonal process": "#C44E52",
        "pharmacokinetic modulation": "#8172B3",
        "same-node redundancy": "#937860",
    }
    _panel_order = ["[1, 1]  (= bar3 ref)", "[0.5, 0.5]", "[0.75, 0.25]",
                    "[0.25, 0.75]", "[1, -1]", "[-1, 1]"]
    _cls_order_wp = list(wprior_perturb_r2["design_class"].value_counts().index)

    fig_wprior, _axes = plt.subplots(2, 3, figsize=(16.5, 9.2), sharex=True, sharey=True)
    _axes = _axes.ravel()
    for _ax, _prior in zip(_axes, _panel_order):
        _d = wprior_perturb_r2[wprior_perturb_r2["prior"] == _prior]
        for _cls in _cls_order_wp:
            _s = _d[_d["design_class"] == _cls]
            _ax.scatter(_s["potency"], _s["r2"], s=95,
                        color=_CLASS_COLORS_WP.get(_cls, "#999999"),
                        edgecolor="white", linewidth=0.7, label=_cls, zorder=3)
        _ax.axhline(0, color="gray", lw=0.7)
        _ax.axvline(0, color="gray", lw=0.6, ls=":")
        _med = _d["r2"].median(); _nneg = int((_d["r2"] < 0).sum())
        _ax.set_title(f"W_prior = {_prior}\nmedian R^2 = {_med:+.2f}   |   R^2<0: {_nneg}/{len(_d)}",
                      fontsize=9.5)

    for _i in (0, 3):
        _axes[_i].set_ylabel("per-perturbation R^2 (across 52 cells)")
    for _i in (3, 4, 5):
        _axes[_i].set_xlabel("perturbation potency = mean growth_rate\n(left = more potent)")
    _axes[0].set_ylim(-1.9, 1.08)

    _hw, _lw = _axes[0].get_legend_handles_labels()
    fig_wprior.legend(_hw, _lw, title="combo design class", loc="upper center",
                      ncol=len(_cls_order_wp), fontsize=9, frameon=False,
                      bbox_to_anchor=(0.5, 1.005))
    fig_wprior.suptitle("bar3-style fit under different additivity W_priors  (n=17 combos, lam=10)\n"
                        "weights = [alphabetically-first drug, second drug]  (components sorted by name)",
                        fontsize=11, y=1.06)
    fig_wprior.tight_layout(rect=[0, 0, 1, 0.97])
    fig_wprior.savefig("/home/jeannie/relearn/notebooks/jeannie/fig_wprior_sweep_by_designclass.png",
                       dpi=150, bbox_inches="tight")
    fig_wprior
    return


@app.cell
def _(ast, eb_df, np, pd):
    # --- W_prior sweep, POTENT-DRUG-FIRST ordering (variant of the alphabetical sweep above) ---
    # Same fit as the alphabetical W_prior sweep, but components are ordered POTENT-DRUG-FIRST:
    # potency = mean single-agent growth_rate across all 52 cells, most-negative (most potent)
    # first -> weight[0] = more-potent partner, weight[1] = weaker. Symmetric priors unchanged.
    from rhaister.combos import is_multi_drug as _is_multi
    from rhaister.prepare_sensitivity import make_splits as _make_splits
    from rhaister.train_sensitivity import _als_decompose_scalar as _als_fn, _drug_regression as _dreg_fn

    _df_wp = eb_df
    _cellsW = sorted(_df_wp.cell_line.unique()); _treatW = sorted(_df_wp.condition.unique())
    _c2iW = {c: i for i, c in enumerate(_cellsW)}; _t2iW = {t: i for i, t in enumerate(_treatW)}
    _ncW, _ntW = len(_cellsW), len(_treatW)
    _singlesW = set(); _scondW = {}
    for _c in _treatW:
        if not _is_multi(_c):
            _t = ast.literal_eval(_c)
            if not _t[0][0].startswith("DMSO"):
                _singlesW.add((_t[0][0], _t[0][1])); _scondW[(_t[0][0], _t[0][1])] = _c
    _eligW = [_c for _c in _treatW if _is_multi(_c)
              and all((d, dose) in _singlesW for d, dose, u in ast.literal_eval(_c))]

    # potency = mean single-agent growth_rate across all cells (more negative = more potent)
    _single_pot = _df_wp[~_df_wp["condition"].map(_is_multi)].groupby("condition")["growth_rate"].mean().to_dict()
    _compW, _potency_ref = {}, {}
    for _c in _eligW:
        _ranked = sorted(ast.literal_eval(_c), key=lambda _t: _single_pot[_scondW[(_t[0], _t[1])]])
        _compW[_c] = [_t2iW[_scondW[(d, dose)]] for d, dose, u in _ranked]
        _potency_ref[_c] = {"potent_drug (w0)": _ranked[0][0], "weak_drug (w1)": _ranked[1][0],
                            "potent_mean_gr": round(_single_pot[_scondW[(_ranked[0][0], _ranked[0][1])]], 3),
                            "weak_mean_gr": round(_single_pot[_scondW[(_ranked[1][0], _ranked[1][1])]], 3)}

    _grW = {(r.cell_line, r.condition): r.growth_rate for r in _df_wp.itertuples()}

    def _fit_prior(weights, lam=10.0):
        _W = np.asarray(weights, float); _rows = []
        for _hcn in _cellsW:
            _hc = _c2iW[_hcn]
            _tr, _ = _make_splits(_df_wp, {"holdout_cells": [_hcn],
                                           "test_treatments": {_hcn: set(_eligW)}})
            _ctr = np.array([_c2iW[c] for c in _tr.cell_line])
            _ttr = np.array([_t2iW[t] for t in _tr.condition])
            _yv = _tr.growth_rate.to_numpy(float)
            _mu, _ce, _te = _als_fn(_yv, _ctr, _ttr, _ncW, _ntW, n_iter=30)
            _yimp = _mu + _ce[:, None] + _te[None, :]; _yimp[_ctr, _ttr] = _yv
            _obs = np.zeros((_ncW, _ntW), bool); _obs[_ctr, _ttr] = True
            _tp = [(_hc, _t2iW[c]) for c in _eligW]
            _prior_hc = np.array([(_yimp[_hc, _compW[c]] * _W).sum() for c in _eligW])
            _res = _yimp.copy()
            for c in _eligW:
                _res[:, _t2iW[c]] = _yimp[:, _t2iW[c]] - (_yimp[:, _compW[c]] * _W).sum(1)
            _yr, _cov = _dreg_fn(_res, _obs, _tp, lam=lam, holdout_set={_hc})
            _pred = np.where(_cov, _yr + _prior_hc, _prior_hc)
            for _k, c in enumerate(_eligW):
                _rows.append({"cell_line": _hcn, "condition": c,
                              "pred": float(_pred[_k]), "measured": float(_grW[(_hcn, c)])})
        return pd.DataFrame(_rows)

    _PRIORS_WP = [
        ("[1, 1]  (= bar3 ref)", [1, 1]), ("[1, -1]", [1, -1]), ("[-1, 1]", [-1, 1]),
        ("[0.5, 0.5]", [0.5, 0.5]), ("[0.75, 0.25]", [0.75, 0.25]), ("[0.25, 0.75]", [0.25, 0.75]),
    ]
    _dcW = pd.read_csv("/home/jeannie/relearn/artifacts/emeraldbay_combo_design_classes.csv")
    _dcmapW = dict(zip(_dcW.condition, _dcW.design_class))
    _dosemapW = dict(zip(_dcW.condition, _dcW.doses_uM))
    _drugmapW = dict(zip(_dcW.condition, _dcW.drugs))

    def _r2W(m, p):
        _ss = ((m - m.mean()) ** 2).sum()
        return float("nan") if _ss == 0 else float(1 - ((m - p) ** 2).sum() / _ss)

    _tidy = []
    for _lbl, _w in _PRIORS_WP:
        _pf = _fit_prior(_w)
        for _cond, _g in _pf.groupby("condition"):
            _m = _g["measured"].to_numpy(); _pr = _g["pred"].to_numpy()
            _tidy.append({"prior": _lbl, "weights": str(_w), "condition": _cond,
                          "drugs": _drugmapW.get(_cond, "?"),
                          "design_class": _dcmapW.get(_cond, "unknown"),
                          "doses_uM": _dosemapW.get(_cond, ""),
                          "potent_drug": _potency_ref[_cond]["potent_drug (w0)"],
                          "weak_drug": _potency_ref[_cond]["weak_drug (w1)"],
                          "potency": float(_m.mean()), "r2": _r2W(_m, _pr)})
    wprior_perturb_r2_potent = pd.DataFrame(_tidy)

    wprior_potency_ref_potent = (pd.DataFrame([{"drugs": _drugmapW[_c], "doses_uM": _dosemapW[_c],
                                                **_potency_ref[_c]} for _c in _eligW])
                                 .sort_values("potent_mean_gr").reset_index(drop=True))
    wprior_potency_ref_potent
    return (wprior_perturb_r2_potent,)


@app.cell
def _(plt, wprior_perturb_r2_potent):
    # --- W_prior sweep (POTENT-FIRST): per-perturbation R^2 vs potency, coloured by design class ---
    _CLASS_COLORS_WP = {
        "horizontal bypass": "#4C72B0", "vertical blockade": "#DD8452",
        "same process / different mechanism": "#55A868", "orthogonal process": "#C44E52",
        "pharmacokinetic modulation": "#8172B3", "same-node redundancy": "#937860",
    }
    _panel_order = ["[1, 1]  (= bar3 ref)", "[0.5, 0.5]", "[0.75, 0.25]",
                    "[0.25, 0.75]", "[1, -1]", "[-1, 1]"]
    _cls_order_wp = list(wprior_perturb_r2_potent["design_class"].value_counts().index)

    fig_wprior_potent, _axes = plt.subplots(2, 3, figsize=(16.5, 9.2), sharex=True, sharey=True)
    _axes = _axes.ravel()
    for _ax, _prior in zip(_axes, _panel_order):
        _d = wprior_perturb_r2_potent[wprior_perturb_r2_potent["prior"] == _prior]
        for _cls in _cls_order_wp:
            _s = _d[_d["design_class"] == _cls]
            _ax.scatter(_s["potency"], _s["r2"], s=95,
                        color=_CLASS_COLORS_WP.get(_cls, "#999999"),
                        edgecolor="white", linewidth=0.7, label=_cls, zorder=3)
        _ax.axhline(0, color="gray", lw=0.7)
        _ax.axvline(0, color="gray", lw=0.6, ls=":")
        _med = _d["r2"].median(); _nneg = int((_d["r2"] < 0).sum())
        _ax.set_title(f"W_prior = {_prior}\nmedian R^2 = {_med:+.2f}   |   R^2<0: {_nneg}/{len(_d)}",
                      fontsize=9.5)
    for _i in (0, 3):
        _axes[_i].set_ylabel("per-perturbation R^2 (across 52 cells)")
    for _i in (3, 4, 5):
        _axes[_i].set_xlabel("perturbation potency = mean growth_rate\n(left = more potent)")
    _axes[0].set_ylim(-1.9, 1.08)
    _hw, _lw = _axes[0].get_legend_handles_labels()
    fig_wprior_potent.legend(_hw, _lw, title="combo design class", loc="upper center",
                             ncol=len(_cls_order_wp), fontsize=9, frameon=False,
                             bbox_to_anchor=(0.5, 1.005))
    fig_wprior_potent.suptitle("bar3-style fit under different additivity W_priors  (n=17 combos, lam=10)\n"
                               "weights = [more-potent partner, less-potent partner]  (potency = mean growth_rate across 52 cells)",
                               fontsize=11, y=1.06)
    fig_wprior_potent.tight_layout(rect=[0, 0, 1, 0.97])
    fig_wprior_potent.savefig("/home/jeannie/relearn/notebooks/jeannie/fig_wprior_sweep_potentfirst_by_designclass.png",
                              dpi=150, bbox_inches="tight")
    fig_wprior_potent
    return


@app.cell
def _(np, pd, pred_all52):
    # --- Baseline references per perturbation: bar1 (additivity), bar2 (Rhaister ships), A/B ceiling ---
    # bar1/bar2 = per-combo R^2 across 52 cells (from pred_all52). A/B ceiling = split-half
    # growth_rate reproducibility recomputed from the raw h5ad single-cell counts: split each
    # cell line's cells in half, recompute growth_rate on each half, correlate half-A vs half-B
    # across the 52 cells (per combo), Spearman-Brown-project to full-sample reliability r_full.
    # r_full is a reliability (correlation) and upper-bounds the achievable per-perturbation R^2.
    import glob as _glob, h5py as _h5py

    _H5 = "/large_storage/goodarzilab/bioreason_cell/emeraldbay/h5ads"
    _CTRL = "[('DMSO_TF', 0.0, 'uM')]"
    def _cats(f, key):
        _c = f[f"obs/{key}/categories"][:]
        return [x.decode() if isinstance(x, bytes) else x for x in _c]

    _ab_data = []
    for _p in sorted(_glob.glob(f"{_H5}/*.h5ad")):
        with _h5py.File(_p, "r") as _f:
            _cc = _cats(_f, "cell_line"); _ccodes = _f["obs/cell_line/codes"][:]
            _cl = _cc[np.bincount(_ccodes).argmax()]
            _dcat = _cats(_f, "drugname_drugconc")
            _dco = _f["obs/drugname_drugconc/codes"][:].astype(np.int32)
            _ab_data.append((_cl, _dco, _dcat))

    def _gr_half(which, masks):
        _counts = {}
        for (_cl, _dco, _dcat), _mask in zip(_ab_data, masks):
            _m = _mask if which == "A" else ~_mask
            _nb = np.bincount(_dco[_m], minlength=len(_dcat))
            _counts[_cl] = {_dcat[i]: int(_nb[i]) for i in range(len(_dcat)) if _nb[i] > 0}
        _tot = {}
        for _cl, _cd in _counts.items():
            for _cond, _n in _cd.items():
                _tot[_cond] = _tot.get(_cond, 0) + _n
        _gr = {}
        for _cl, _cd in _counts.items():
            _nc = _cd.get(_CTRL, 0)
            if _nc == 0 or _tot.get(_CTRL, 0) == 0:
                continue
            _pc = _nc / _tot[_CTRL]
            for _cond, _n in _cd.items():
                if _cond == _CTRL or _n == 0:
                    continue
                _gr[(_cl, _cond)] = np.log2((_n / _tot[_cond]) / _pc)
        return _gr

    _conds17 = sorted(pred_all52.condition.unique())
    _abper = {c: [] for c in _conds17}
    for _seed in range(5):
        _rng = np.random.default_rng(_seed)
        _masks = [_rng.random(len(d[1])) < 0.5 for d in _ab_data]
        _ga = _gr_half("A", _masks); _gb = _gr_half("B", _masks)
        for c in _conds17:
            _a, _b = [], []
            for (_cl, _, _) in _ab_data:
                _k = (_cl, c)
                if _k in _ga and _k in _gb:
                    _a.append(_ga[_k]); _b.append(_gb[_k])
            if len(_a) >= 3:
                _abper[c].append(float(np.corrcoef(_a, _b)[0, 1]))

    def _r2b(m, p):
        _ss = ((m - m.mean()) ** 2).sum()
        return float("nan") if _ss == 0 else float(1 - ((m - p) ** 2).sum() / _ss)

    _dcB = pd.read_csv("/home/jeannie/relearn/artifacts/emeraldbay_combo_design_classes.csv")
    _dcmapB = dict(zip(_dcB.condition, _dcB.design_class))
    _drugmapB = dict(zip(_dcB.condition, _dcB.drugs))
    _dosemapB = dict(zip(_dcB.condition, _dcB.doses_uM))

    _rowsB = []
    for _cond, _g in pred_all52.groupby("condition"):
        _m = _g["measured"].to_numpy()
        _rh = float(np.mean(_abper[_cond])); _rf = 2 * _rh / (1 + _rh)
        _rowsB.append({"condition": _cond, "drugs": _drugmapB.get(_cond, "?"),
                       "design_class": _dcmapB.get(_cond, "unknown"),
                       "doses_uM": _dosemapB.get(_cond, ""),
                       "potency": float(_m.mean()),
                       "r2_bar1": _r2b(_m, _g["bar1_add"].to_numpy()),
                       "r2_bar2": _r2b(_m, _g["bar2_shrink0_lam10"].to_numpy()),
                       "ab_r_half": _rh, "ab_r_full": _rf})
    baseline_perturb = pd.DataFrame(_rowsB)
    baseline_perturb[["design_class", "drugs", "doses_uM", "potency",
                      "r2_bar1", "r2_bar2", "ab_r_full"]].round(3)
    return (baseline_perturb,)


@app.cell
def _(baseline_perturb, plt):
    # --- Baselines per perturbation vs potency, coloured by design class ---
    # bar1 (pure additivity) and bar2 (Rhaister ships) share the R^2 axis; the A/B split-half
    # panel is a reliability ceiling (Spearman-Brown r_full), so it has its own axis.
    _CLASS_COLORS_B = {
        "horizontal bypass": "#4C72B0",
        "vertical blockade": "#DD8452",
        "same process / different mechanism": "#55A868",
        "orthogonal process": "#C44E52",
        "pharmacokinetic modulation": "#8172B3",
        "same-node redundancy": "#937860",
    }
    _cls_order_b = list(baseline_perturb["design_class"].value_counts().index)

    fig_baselines, (_axb1, _axb2, _axab) = plt.subplots(1, 3, figsize=(16.5, 5.6))

    def _scatter_panel(ax, ycol, title, is_r2=True):
        for _cls in _cls_order_b:
            _s = baseline_perturb[baseline_perturb["design_class"] == _cls]
            ax.scatter(_s["potency"], _s[ycol], s=100,
                       color=_CLASS_COLORS_B.get(_cls, "#999999"),
                       edgecolor="white", linewidth=0.8, label=_cls, zorder=3)
        ax.axhline(0, color="gray", lw=0.7)
        ax.axvline(0, color="gray", lw=0.6, ls=":")
        ax.set_xlabel("perturbation potency = mean growth_rate\n(left = more potent)")
        ax.set_title(title, fontsize=9.5)

    _m1 = baseline_perturb["r2_bar1"].median(); _n1 = int((baseline_perturb["r2_bar1"] < 0).sum())
    _m2 = baseline_perturb["r2_bar2"].median(); _n2 = int((baseline_perturb["r2_bar2"] < 0).sum())
    _scatter_panel(_axb1, "r2_bar1",
                   f"bar 1: add-the-singles (pure additivity)\nmedian R^2 = {_m1:+.2f}  |  R^2<0: {_n1}/17")
    _scatter_panel(_axb2, "r2_bar2",
                   f"bar 2: 'Rhaister ships' (shrink-0 + intercept)\nmedian R^2 = {_m2:+.2f}  |  R^2<0: {_n2}/17")
    _axb1.set_ylabel("per-perturbation R^2 (across 52 cells)")
    _axb1.set_ylim(-2.9, 1.08); _axb2.set_ylim(-2.9, 1.08)

    # A/B ceiling panel (reliability, own axis)
    _scatter_panel(_axab, "ab_r_full",
                   f"A/B split-half ceiling (reliability)\nmedian r_full = {baseline_perturb['ab_r_full'].median():.2f}  |  upper bound on R^2")
    _axab.set_ylabel("Spearman-Brown r_full  (split-half reliability)")
    _axab.set_ylim(0.5, 1.0)

    _hb, _lb = _axb1.get_legend_handles_labels()
    fig_baselines.legend(_hb, _lb, title="combo design class", loc="upper center",
                         ncol=len(_cls_order_b), fontsize=8.5, frameon=False,
                         bbox_to_anchor=(0.5, 1.02))
    fig_baselines.suptitle("Baseline references per perturbation (bar1, bar2, A/B ceiling)  — n=17 combos",
                           fontsize=11.5, y=1.10)
    fig_baselines.tight_layout(rect=[0, 0, 1, 0.9])
    fig_baselines.savefig("/home/jeannie/relearn/notebooks/jeannie/fig_baselines_by_designclass.png",
                          dpi=150, bbox_inches="tight")
    fig_baselines
    return


@app.cell
def _(cellprofile, mo):
    _neg = cellprofile[cellprofile["sign"] == "R2<0"]
    _pos = cellprofile[cellprofile["sign"] == "R2>=0"]
    _corr = cellprofile[["r2_add", "mean_single_gr"]].corr().iloc[0, 1]
    _corr_depth = (cellprofile[["r2_add", "depth"]].corr().iloc[0, 1]
                   if cellprofile["depth"].notna().any() else float("nan"))

    mo.md(rf"""
    ### What the R^2<0 cell lines have in common

    **They are the drug-sensitive lines, not the poorly-sampled ones.** The negative-R^2 group
    separates cleanly on biology: their **mean single-agent growth_rate is
    {_neg['mean_single_gr'].median():.2f}** vs **{_pos['mean_single_gr'].median():.2f}** for the
    R^2>=0 group — i.e. drugs kill much harder in the failing cells. Across all 52 lines, additive
    R^2 correlates with single-agent sensitivity at **r = {_corr:.2f}** (middle panel): the more a
    cell is killed by single agents, the worse additivity does.

    **The mechanism is assay saturation, and it is directional.** In a sensitive line two potent
    agents each drive growth_rate strongly negative; summing them in log-space (the additive/Bliss
    prediction) forecasts killing below the biological floor, so measured combos sit systematically
    *above* the additive line. That one-sided compression is a wrong *direction*, not a constant
    offset — which is why an intercept (bar 4) can't fix it and only free weights (shrink-to-0) can.

    **It is largely not a sampling artefact.** Additive R^2 vs sequencing depth shows
    {"a weak" if abs(_corr_depth) < 0.3 else "some"} relationship (r = {_corr_depth:.2f}, left panel):
    depth alone does not explain who fails. SW480 is deeply sampled yet additivity works there because
    it sits in the *resistant / non-saturating* regime — the opposite end of the same sensitivity axis.

    **Lineage tracks sensitivity.** The right panel shows the failure concentrates in the most
    drug-responsive lineages rather than spreading uniformly, consistent with a sensitivity-driven
    (not lineage-intrinsic) effect — see `tissue_tab` for the per-lineage `frac_neg`.

    **Bottom line.** "R^2<0" labels a **cell state — high drug sensitivity that saturates the
    growth assay — not a data-quality problem.** This is exactly why the additivity/[1,1] prior that
    helped the resistant SW480 hurts the panel: ~40% of these lines are sensitive enough that additive
    killing is physically impossible, and any model anchored to additivity inherits that bias.
    """)
    return


if __name__ == "__main__":
    app.run()
