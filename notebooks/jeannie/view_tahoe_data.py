import marimo

__generated_with = "0.23.16"
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
    return


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

    return (mo,)


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

    fig_bars, (axA, axB) = _plt.subplots(1, 2, figsize=(11, 4.4))
    _w = 0.38
    axA.bar(_x - _w/2, _r2, _w, label="R^2", color="#4C72B0")
    axA.bar(_x + _w/2, _pe, _w, label="Pearson", color="#DD8452")
    axA.axhline(bars52_ceiling, ls="--", color="gray", lw=1)
    axA.text(0.0, bars52_ceiling + 0.015, f"A/B sampling ceiling {bars52_ceiling:.2f}", ha="left", fontsize=8, color="gray")
    axA.axhline(0, color="k", lw=0.6)
    axA.set_xticks(_x); axA.set_xticklabels(_labels, fontsize=7.5)
    axA.set_ylabel("pooled score (884 combos)"); axA.set_ylim(-0.15, 1.0)
    axA.legend(fontsize=8, loc="center right"); axA.set_title("Pooled across 52 cells (lambda=10)", fontsize=10)

    _bp = axB.boxplot(_pcv, showfliers=False, positions=_x, widths=0.5, patch_artist=True)
    for _p in _bp["boxes"]:
        _p.set_facecolor("#B0C4DE")
    for _i in range(4):
        axB.plot(_x[_i], _pcv[_i][_swi], marker="*", color="red", ms=13, zorder=6)
    axB.axhline(0, color="k", lw=0.6); axB.set_ylim(-1.2, 1.0)
    axB.set_xticks(_x); axB.set_xticklabels(_labels, fontsize=7.5)
    axB.set_ylabel("per-cell R^2"); axB.set_title("Per-cell spread (red * = SW480)", fontsize=10)

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
def _(bars52):
    # --- Scatter: per-cell R^2, bar 3 vs bar 4 (each dot = one held-out cell line) ---
    import matplotlib.pyplot as _plts
    import numpy as _nps

    _b3 = _nps.array(bars52["per_cell_R2"]["bar3"])
    _b4 = _nps.array(bars52["per_cell_R2"]["bar4"])
    _swi = bars52["sw480_idx"]
    _lo, _hi = -1.5, 1.0
    _b3c, _b4c = _nps.clip(_b3, _lo, _hi), _nps.clip(_b4, _lo, _hi)
    _clip = int(((_b3 < _lo) | (_b4 < _lo)).sum())

    # colour by sign quadrant
    _col = _nps.where((_b3 > 0) & (_b4 > 0), "#2a9d8f",
            _nps.where((_b3 < 0) & (_b4 < 0), "#e76f51", "#9aa0a6"))

    fig_scatter, ax = _plts.subplots(figsize=(6.2, 6.2))
    ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.7)
    ax.plot([_lo, _hi], [_lo, _hi], ls="--", color="k", lw=0.8, label="bar 3 = bar 4")
    ax.scatter(_b3c, _b4c, c=_col, s=45, edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter([_b3c[_swi]], [_b4c[_swi]], marker="*", s=320, color="red",
               edgecolor="black", linewidth=0.5, zorder=5, label="SW480")
    ax.set_xlim(_lo, _hi); ax.set_ylim(_lo, _hi)
    ax.set_xlabel("bar 3 per-cell R^2   ([1,1], no intercept)")
    ax.set_ylabel("bar 4 per-cell R^2   ([1,1] + intercept)")
    _np3 = int((_b3 > 0).sum()); _np4 = int((_b4 > 0).sum()); _better = int((_b4 > _b3).sum())
    ax.set_title(f"Per-cell R^2: bar3 vs bar4\n"
                 f"positive R^2: bar3 {_np3}/52, bar4 {_np4}/52   |   intercept helps (bar4>bar3): {_better}/52",
                 fontsize=9)
    ax.text(_lo + 0.03, _lo + 0.03, f"{_clip} cell(s) clipped to floor (R^2 < {_lo})",
            fontsize=7.5, color="gray")
    ax.text(0.55, -0.35, "both R^2 > 0", color="#2a9d8f", fontsize=8, ha="center")
    ax.text(-0.75, -1.25, "both R^2 < 0", color="#e76f51", fontsize=8, ha="center")
    ax.legend(fontsize=8, loc="upper left")
    fig_scatter.tight_layout()
    fig_scatter.savefig("/home/jeannie/relearn/notebooks/jeannie/fig_bar3_vs_bar4.png", dpi=150, bbox_inches="tight")
    fig_scatter
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


if __name__ == "__main__":
    app.run()
