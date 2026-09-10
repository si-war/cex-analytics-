"""
Reusable analysis + prediction logic for the CEXInsight app (testable without Streamlit).
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as st

from pipeline import CONT_FEATS, TARGET

EDU_ORDER_MAIN = [10, 11, 12, 13, 14, 15, 16, 0]
EDU_LABELS = {
    0: "Never attended", 10: "1st-8th grade", 11: "9th-12th no diploma",
    12: "HS graduate", 13: "Some college", 14: "Associate degree",
    15: "Bachelor degree", 16: "Advanced degree",
}


def predict_row(row_cont, row_enc, model, scaler):
    Xc = pd.DataFrame(row_cont, columns=CONT_FEATS)
    X = np.hstack([scaler.transform(Xc), row_enc])
    return max(0, float(model.predict(X)[0]))


def contribution_df(model, scaler, row_cont, row_enc, df, nice_names, top_n=8):
    """Plain-language 'why this income' table: your value vs typical household."""
    feat_names = CONT_FEATS + ["region_2", "region_3", "region_4", "sex_male", "edu_ordinal"]
    importances = np.asarray(model.feature_importances_)
    top_idx = np.argsort(importances)[::-1][:top_n]

    enc_names = ["region_2", "region_3", "region_4", "sex_male", "edu_ordinal"]
    typical = np.concatenate([
        df[CONT_FEATS].median().to_numpy(),
        np.array([
            1.0 if (df["REGION"] == 2).mean() >= 0.5 else 0.0,
            1.0 if (df["REGION"] == 3).mean() >= 0.5 else 0.0,
            1.0 if (df["REGION"] == 4).mean() >= 0.5 else 0.0,
            df["SEX_REF"].median() == 1,
            df["HIGH_EDU"].median(),
        ], dtype=float),
    ])
    your = np.concatenate([row_cont[0], row_enc[0]])

    rows = []
    for i in top_idx:
        med = typical[i]
        if abs(med) > 1e-9:
            pct = (your[i] - med) / abs(med) * 100
        else:
            pct = 0.0
        if abs(pct) < 2:
            status = "Similar to a typical household"
        elif pct > 0:
            status = "Above typical — pushes prediction up"
        else:
            status = "Below typical — pulls prediction down"
        rows.append({
            "Feature": nice_names.get(feat_names[i], feat_names[i]),
            "Your household": f"{your[i]:,.1f}",
            "Typical household": f"{med:,.1f}",
            "Difference": f"{your[i] - med:+,.1f}",
            "What it means": status,
        })
    return pd.DataFrame(rows)


def engel_table(df, income="FINCBTXM_M"):
    """Food + housing share of spending by income quarter (Engel's law check)."""
    d = df.copy()
    d["food_share"] = d["food_total_m"] / d["TOTEXPPQ_M"].clip(lower=1)
    d["housing_share"] = d["HOUSPQ_M"] / d["TOTEXPPQ_M"].clip(lower=1)
    d["iq"] = pd.qcut(d[income].clip(lower=0), 4, labels=["Poorest 25%", "Lower-middle", "Upper-middle", "Richest 25%"])
    t = d.groupby("iq", observed=True).agg(
        Median_income=(income, "median"),
        Food_share=("food_share", "median"),
        Housing_share=("housing_share", "median"),
        Households=(income, "count"),
    )
    t["Median_income"] = t["Median_income"].map("${:,.0f}".format)
    t["Food_share"] = t["Food_share"].map("{:.1%}".format)
    t["Housing_share"] = t["Housing_share"].map("{:.1%}".format)
    return t


def budget_shares_by_education(df):
    share_cols = ["share_food", "share_housing", "share_transport", "share_health",
                  "share_entertainment", "share_apparel", "share_alcohol"]
    d = df[df["HIGH_EDU"].isin(EDU_ORDER_MAIN)].copy()
    d["edu_label"] = d["HIGH_EDU"].map(EDU_LABELS)
    d["edu_label"] = pd.Categorical(d["edu_label"], categories=[EDU_LABELS[e] for e in EDU_ORDER_MAIN][::-1], ordered=True)
    return d.groupby("edu_label", observed=True)[share_cols].mean()


def education_income_stats(df):
    """ANOVA across education levels + OLS net premium table (client-friendly)."""
    order = [e for e in EDU_ORDER_MAIN if e in df["HIGH_EDU"].unique()]
    d = df[df["HIGH_EDU"].isin(order)].copy()
    d["edu_label"] = d["HIGH_EDU"].map(EDU_LABELS)
    cats = [e for e in EDU_ORDER_MAIN if e in order]
    d["edu_label"] = pd.Categorical(d["edu_label"], categories=[EDU_LABELS[e] for e in cats], ordered=True)

    groups = [g["FINCBTXM_M"].dropna().values for _, g in d.groupby("edu_label", observed=True) if len(g) > 5]
    f_stat, p_val = st.f_oneway(*groups)

    # OLS net premium: income ~ education + age + family + children + sex (HC3)
    d = d.copy()
    d["age"] = d["AGE_REF"]
    d["fam_size"] = d["FAM_SIZE"]
    d["perslt18"] = d["PERSLT18"]
    d["sex_male"] = (d["SEX_REF"] == 1).astype(int)
    d["income_k"] = d["FINCBTXM_M"] / 1000
    ref = 12
    terms = [f"edu_{e}" for e in order if e != ref]
    for e in order:
        if e != ref:
            d[f"edu_{e}"] = (d["HIGH_EDU"] == e).astype(int)
    formula = f"income_k ~ {' + '.join(terms)} + age + fam_size + perslt18 + sex_male"
    ols = smf.ols(formula, data=d).fit(cov_type="HC3")

    rows = []
    for e in [e for e in order if e != ref]:
        coef = ols.params[f"edu_{e}"]
        p = ols.pvalues[f"edu_{e}"]
        rows.append({
            "Education level": EDU_LABELS[e],
            "Extra monthly income vs HS graduate": f"${coef * 1000:+,.0f}",
            "p-value": f"{p:.4f}",
            "Statistically significant": "Yes" if p < 0.05 else "No",
        })
    return d, rows, f_stat, p_val, ols


def gender_interaction_test(df):
    """Does the education premium differ between men and women? (interaction model)"""
    order = [e for e in EDU_ORDER_MAIN if e in df["HIGH_EDU"].unique()]
    d = df[df["HIGH_EDU"].isin(order)].copy()
    d["age"] = d["AGE_REF"]
    d["fam_size"] = d["FAM_SIZE"]
    d["perslt18"] = d["PERSLT18"]
    d["male"] = (d["SEX_REF"] == 1).astype(int)
    d["income_m"] = d["FINCBTXM_M"] / 1000
    ref = 12
    terms = [f"edu_{e}" for e in order if e != ref]
    for e in order:
        if e != ref:
            d[f"edu_{e}"] = (d["HIGH_EDU"] == e).astype(int)

    formula = ("income_m ~ " + " + ".join(terms)
               + " + age + fam_size + perslt18 + male"
               + " + " + " + ".join(f"{t}:male" for t in terms))
    ols = smf.ols(formula, data=d).fit(cov_type="HC3")
    f_test_expr = " = ".join(f"{t}:male" for t in terms) + " = 0"
    ft = ols.f_test(f_test_expr)

    rows = []
    for e in [e for e in order if e != ref]:
        var, inter = f"edu_{e}", f"edu_{e}:male"
        coef_f = ols.params[var]
        inter_coef = ols.params.get(inter, 0)
        coef_m = coef_f + inter_coef
        p = ols.pvalues.get(inter, 1.0)
        rows.append({
            "Education level": EDU_LABELS[e],
            "Premium for men": f"${coef_m * 1000:+,.0f}",
            "Premium for women": f"${coef_f * 1000:+,.0f}",
            "Difference (men-women)": f"${inter_coef * 1000:+,.0f}",
            "p-value": f"{p:.4f}",
            "Significant gap": "Yes" if p < 0.05 else "No",
        })
    return pd.DataFrame(rows), float(ft.pvalue), ols


def education_levels_data(df):
    order = [e for e in EDU_ORDER_MAIN if e in df["HIGH_EDU"].unique()]
    d = df[df["HIGH_EDU"].isin(order)].copy()
    d["edu_label"] = d["HIGH_EDU"].map(EDU_LABELS)
    cats = [e for e in EDU_ORDER_MAIN if e in order]
    d["edu_label"] = pd.Categorical(d["edu_label"], categories=[EDU_LABELS[e] for e in cats], ordered=True)
    return d, cats


# ───────────────────────── Uncertainty (prediction honesty) ─────────────────────────

def holdout_eval(df, model, test_size=0.2, random_state=42):
    """Hold-out predictions from the CURRENT model → scatter + residual distribution.

    Uses the same split convention as the training pipeline so the numbers line up
    with the stored metrics. Returns a DataFrame of actual / predicted / residual.
    """
    from pipeline import build_matrix
    from sklearn.model_selection import train_test_split

    y = df[TARGET]
    idx = np.arange(len(df))
    _, idx_te = train_test_split(idx, test_size=test_size, random_state=random_state)
    _, X_cs, X_e, _, _ = build_matrix(df)
    X_full = np.hstack([X_cs, X_e])
    pred = model.predict(X_full[idx_te])

    out = pd.DataFrame({
        "actual": y.to_numpy()[idx_te],
        "predicted": pred,
        "residual": y.to_numpy()[idx_te] - pred,
    })
    out["edu_label"] = df["HIGH_EDU"].to_numpy()[idx_te].astype(int)
    out["education"] = out["edu_label"].map(EDU_LABELS)
    return out


def residual_bands(resid):
    """Empirical ±bands around a prediction, from the actual spread of hold-out errors."""
    return {
        68: (float(np.quantile(resid, 0.16)), float(np.quantile(resid, 0.84))),
        95: (float(np.quantile(resid, 0.025)), float(np.quantile(resid, 0.975))),
    }


def interval_text(pred, resid, bands=(68, 68)):
    lo, hi = residual_bands(resid)[bands]
    return f"Range for the middle {bands}% of households: **${max(0, pred + lo):,.0f} – ${pred + hi:,.0f}/month**"


# ───────────────────────── Percentiles (population context) ─────────────────────────

def income_percentile(income, df):
    """What % of households earn less than `income`?"""
    inc = df[TARGET].dropna()
    return float((inc < income).mean()) * 100


def income_for_percentile(p, df):
    """Income that separates the bottom `p`% from the top."""
    return float(np.quantile(df[TARGET].dropna().to_numpy(), p / 100))


# ───────────────────────── Budget simulator (rules are interpretive) ─────────────────────────

def budget_vs_rule(income_m, totexp_pq, hous_pq, food_home_pq, food_away_pq,
                   trans_pq, health_pq, educa_pq, enter_pq, app_pq, alc_pq, toba_pq):
    """Your household's budget split vs the 50/30/20 rule + the 30% housing guideline.

    Inputs are quarterly (PQ) amounts from the predict form; income is monthly.
    'Savings' is spending not observed by the survey → inferred as income − total spend.
    Returns (rows_for_chart, stats_dict).
    """
    needs_m = (hous_pq + food_home_pq + food_away_pq + trans_pq + health_pq) / 3
    wants_m = (educa_pq + enter_pq + app_pq + alc_pq + toba_pq) / 3
    total_m = totexp_pq / 3
    savings_m = income_m - total_m

    needs_s = needs_m / income_m if income_m > 0 else 0.0
    wants_s = wants_m / income_m if income_m > 0 else 0.0
    savings_s = max(0.0, savings_m) / income_m if income_m > 0 else 0.0
    housing_s = hous_pq / 3 / income_m if income_m > 0 else 0.0
    food_s = (food_home_pq + food_away_pq) / 3 / income_m if income_m > 0 else 0.0

    rows = pd.DataFrame({
        "Bucket": ["Needs (housing, food, transport, health)", "Wants (leisure, clothing, education…)", "Savings (income − spending)"],
        "Your household": [needs_s, wants_s, savings_s],
        "50/30/20 guideline": [0.50, 0.30, 0.20],
    })

    stats = {
        "housing_share": housing_s,
        "food_share": food_s,
        "savings_raw": savings_m,
        "housing_ok": housing_s <= 0.30,
        "food_note": "food takes a smaller share as income rises (Engel's law)",
    }
    return rows, stats


def engel_curve(df, edu_code=None, n_bins=8):
    """Average food + housing budget shares across income groups.

    edu_code=None → all education levels combined. Confirms Engel's law visually:
    food's share falls as income rises.
    """
    order = [e for e in EDU_ORDER_MAIN if e in df["HIGH_EDU"].unique()]
    d = df[df["HIGH_EDU"].isin(order)].copy()
    if edu_code is not None:
        d = d[d["HIGH_EDU"] == edu_code]
    d = d[d[TARGET] > 0]
    d["food_share"] = d["food_total_m"] / d["TOTEXPPQ_M"].clip(lower=1)
    d["housing_share"] = d["HOUSPQ_M"] / d["TOTEXPPQ_M"].clip(lower=1)

    if len(d) < n_bins * 10:
        n_bins = max(2, len(d) // 10)
    if n_bins < 2 or len(d) < 20:
        return pd.DataFrame()

    d["bin"] = pd.qcut(d[TARGET], n_bins, labels=False, duplicates="drop")
    t = d.groupby("bin", observed=True).agg(
        median_income=(TARGET, "median"),
        food_share=("food_share", "mean"),
        housing_share=("housing_share", "mean"),
        n=(TARGET, "size"),
    )
    t.index = t["median_income"].map("${:,.0f}".format)
    return t


def rule_summary(df):
    """What the economic rules of thumb add (US-12c finding, computed live on the data).

    Returns the key numbers that let us say: the rules are *real* in the data
    (Engel's law + 30% housing rule hold), but they are *interpretive* — weak on their
    own, and only slightly enriching the machine-learning model.
    """
    d = df[df[TARGET] > 0].copy()
    d["food_share"] = d["food_total_m"] / d["TOTEXPPQ_M"].clip(lower=1)
    d["housing_share"] = d["HOUSPQ_M"] / d["TOTEXPPQ_M"].clip(lower=1)

    q1, q4 = d[TARGET].quantile([0.25, 0.75])
    food_q1 = d.loc[d[TARGET] <= q1, "food_share"].median()
    food_q4 = d.loc[d[TARGET] >= q4, "food_share"].median()
    hous_q1 = d.loc[d[TARGET] <= q1, "housing_share"].median()
    hous_q4 = d.loc[d[TARGET] >= q4, "housing_share"].median()

    pct_over30 = float((d["housing_share"] > 0.30).mean())

    rho_food = st.spearmanr(d["food_share"], d[TARGET]).statistic
    rho_hous = st.spearmanr(d["housing_share"], d[TARGET]).statistic
    rho_spend = st.spearmanr(d["TOTEXPPQ_M"], d[TARGET]).statistic

    return {
        "food_q1": float(food_q1), "food_q4": float(food_q4),
        "housing_q1": float(hous_q1), "housing_q4": float(hous_q4),
        "pct_over30": pct_over30,
        "rho_food": float(rho_food), "rho_housing": float(rho_hous),
        "rho_spend": float(rho_spend),
    }


def typical_inputs(df, edu_code):
    """Typical household profile + spending for an education level (monthly medians → PQ)."""
    g = df[df["HIGH_EDU"] == edu_code]
    if len(g) == 0:
        g = df
    cols = {
        "totexp_pq": "TOTEXPPQ_M", "hous_pq": "HOUSPQ_M", "shelter_pq": "SHELTPQ_M",
        "trans_pq": "TRANSPQ_M", "health_pq": "HEALTHPQ_M", "educa_pq": "EDUCAPQ_M",
        "enter_pq": "ENTERTPQ_M", "app_pq": "APPARPQ_M", "alc_pq": "ALCBEVPQ_M",
        "toba_pq": "TOBACCPQ_M", "food_home_pq": "food_home_m", "food_away_pq": "food_away_m",
    }
    out = {
        "age": int(g["AGE_REF"].median()),
        "sex": int(g["SEX_REF"].mode().iloc[0]),
        "region": int(g["REGION"].mode().iloc[0]),
        "fam_size": int(g["FAM_SIZE"].median()),
        "perslt18": int(g["PERSLT18"].median()),
        "no_earnr": int(g["NO_EARNR"].median()),
    }
    for pq, mcol in cols.items():
        out[pq] = float(g[mcol].median()) * 3
    return out


def predict_from_education(df, model, scaler, edu_code, **overrides):
    """Predicted monthly income for an education level, everything else typical."""
    from preprocessing import compute_features
    inp = typical_inputs(df, edu_code)
    inp.update(overrides)
    row_cont, row_enc = compute_features(
        inp["age"], inp["sex"], inp["region"], inp["fam_size"], inp["perslt18"],
        inp["no_earnr"], edu_code, inp["totexp_pq"], inp["hous_pq"], inp["shelter_pq"],
        inp["trans_pq"], inp["health_pq"], inp["educa_pq"], inp["enter_pq"],
        inp["app_pq"], inp["alc_pq"], inp["toba_pq"], inp["food_home_pq"],
        inp["food_away_pq"],
    )
    return predict_row(row_cont, row_enc, model, scaler)


def education_ladder(df, model, scaler, highlight=None):
    """Predicted income for every education level (typical profile for each)."""
    order = [e for e in EDU_ORDER_MAIN if e in df["HIGH_EDU"].unique()]
    rows = []
    for e in order:
        pred = predict_from_education(df, model, scaler, e)
        actual = df.loc[df["HIGH_EDU"] == e, TARGET].median()
        rows.append({
            "edu_code": e,
            "Education level": EDU_LABELS[e],
            "Predicted monthly income": round(pred),
            "Actual median income": round(actual),
            "highlight": e == highlight,
        })
    return pd.DataFrame(rows)