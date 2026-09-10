"""
CEX pipeline: turn raw CEX interview files (fmli*.csv) into a clean, feature-engineered
dataset, then train a transparent Gradient Boosting income model with honest metrics.

This module is shared by the notebook and the Streamlit app so the numbers always agree.
"""
import os
import glob
import io
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CONT_FEATS = [
    "TOTEXPPQ_M", "food_total_m", "food_home_m", "food_away_m",
    "HOUSPQ_M", "SHELTPQ_M", "TRANSPQ_M", "HEALTHPQ_M", "EDUCAPQ_M",
    "ENTERTPQ_M", "APPARPQ_M", "ALCBEVPQ_M", "TOBACCPQ_M",
    "AGE_REF", "FAM_SIZE", "PERSLT18", "NO_EARNR",
    "share_food", "share_food_home", "share_food_away", "share_housing",
    "share_shelter", "share_transport", "share_health", "share_education",
    "share_entertainment", "share_apparel", "share_alcohol", "share_tobacco",
    "share_others", "oecd_scale", "percapita_exp_m",
]
ENCODED_FEATS = ["region_2", "region_3", "region_4", "sex_male", "edu_ordinal"]
TARGET = "FINCBTXM_M"

COLS = [
    "CUID", "QINTRVMO", "QINTRVYR",
    "FINCBTXM",
    "EDUC_REF", "HIGH_EDU", "AGE_REF", "SEX_REF", "REGION", "BLS_URBN",
    "FAM_SIZE", "FAM_TYPE", "PERSLT18", "NO_EARNR", "MARITAL1",
    "TOTEXPPQ",
    "FDHOMEPQ", "GROCERPQ", "FDAWAYPQ",
    "HOUSPQ", "SHELTPQ", "TRANSPQ", "HEALTHPQ", "EDUCAPQ",
    "ENTERTPQ", "APPARPQ", "ALCBEVPQ", "TOBACCPQ",
]


def _open_csv(f):
    """Accept either a file path or an in-memory uploaded file object."""
    if hasattr(f, "getvalue"):
        return io.BytesIO(f.getvalue())
    return f


def _filename(f):
    return os.path.basename(f.name if hasattr(f, "name") else f)


def _quarter(name):
    return _filename(name)[4:7]


def _quarter_label(code):
    """'244' -> '2024 Q4'; handles the 'fmli241x' convention (x = Q1)."""
    if len(code) < 3:
        return code
    year = "20" + code[:2]
    qd = code[2]
    q = 1 if not qd.isdigit() or qd == "0" else int(qd)
    return f"{year} Q{q}"


def process_cex_files(fmli_files):
    """Replicates the notebook's US-03..US-06: read, dedupe, clean, engineer features.

    Accepts file paths or in-memory uploaded file objects (Streamlit).
    """
    if not fmli_files:
        raise ValueError("No fmli*.csv files provided.")

    frames = []
    for f in fmli_files:
        buf = _open_csv(f)
        cols = pd.read_csv(buf, nrows=0).columns
        use = [c for c in COLS if c in cols]
        if hasattr(buf, "seek"):
            buf.seek(0)
        d = pd.read_csv(buf, usecols=use, low_memory=False)
        d["quarter"] = _quarter(f)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)

    # --- cleaning (US-05) ---
    df_clean = df.sort_values("quarter").drop_duplicates(subset="CUID", keep="last")
    n_unique = len(df_clean)

    for col in ["FINCBTXM", "TOTEXPPQ"] + [c for c in df_clean.columns if c.endswith("PQ")]:
        df_clean[col] = df_clean[col].clip(lower=0)
    df_clean["REGION"] = df_clean["REGION"].fillna(df_clean["REGION"].mode()[0]).astype(int)

    outlier_cols = ["FINCBTXM", "TOTEXPPQ"] + [c for c in df_clean.columns if c.endswith("PQ")]
    n_capped = 0
    for col in outlier_cols:
        q1, q3 = df_clean[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_capped += int((df_clean[col] > hi).sum()) + int((df_clean[col] < lo).sum())
        df_clean[col] = df_clean[col].clip(lo, hi)

    # --- feature engineering (US-06) ---
    df_clean["food_home_pq"] = df_clean["FDHOMEPQ"].fillna(0) + df_clean["GROCERPQ"].fillna(0)
    df_clean["food_away_pq"] = df_clean["FDAWAYPQ"]
    df_clean["food_total_pq"] = df_clean["food_home_pq"] + df_clean["food_away_pq"]

    df_clean["TOTEXPPQ_M"] = df_clean["TOTEXPPQ"] / 3
    monthly_map = {"food_total_pq": "food_total_m", "food_home_pq": "food_home_m",
                   "food_away_pq": "food_away_m", "HOUSPQ": "HOUSPQ_M", "SHELTPQ": "SHELTPQ_M",
                   "TRANSPQ": "TRANSPQ_M", "HEALTHPQ": "HEALTHPQ_M", "EDUCAPQ": "EDUCAPQ_M",
                   "ENTERTPQ": "ENTERTPQ_M", "APPARPQ": "APPARPQ_M",
                   "ALCBEVPQ": "ALCBEVPQ_M", "TOBACCPQ": "TOBACCPQ_M"}
    for c, m in monthly_map.items():
        df_clean[m] = df_clean[c] / 3

    df_clean["FINCBTXM_M"] = df_clean["FINCBTXM"] / 12  # FINCBTXM is annual

    tot_m = df_clean["TOTEXPPQ_M"]
    share_map = {"food_total_pq": "share_food", "food_home_pq": "share_food_home",
                 "food_away_pq": "share_food_away", "HOUSPQ": "share_housing",
                 "SHELTPQ": "share_shelter", "TRANSPQ": "share_transport",
                 "HEALTHPQ": "share_health", "EDUCAPQ": "share_education",
                 "ENTERTPQ": "share_entertainment", "APPARPQ": "share_apparel",
                 "ALCBEVPQ": "share_alcohol", "TOBACCPQ": "share_tobacco"}
    for c, s in share_map.items():
        df_clean[s] = ((df_clean[c] / 3) / tot_m).replace([np.inf, -np.inf], np.nan).fillna(0)

    top_level = ["share_food", "share_housing", "share_transport", "share_health",
                 "share_education", "share_entertainment", "share_apparel",
                 "share_alcohol", "share_tobacco"]
    df_clean["share_others"] = (1 - df_clean[top_level].sum(axis=1)).clip(lower=0)

    adults = (df_clean["FAM_SIZE"] - df_clean["PERSLT18"]).clip(lower=1)
    children = df_clean["PERSLT18"].clip(lower=0)
    df_clean["oecd_scale"] = 1 + 0.7 * (adults - 1) + 0.5 * children
    df_clean["percapita_exp_m"] = tot_m / df_clean["oecd_scale"]

    report = {
        "raw_rows": n_raw,
        "unique_households": n_unique,
        "nearest_quarter": _quarter_label(_quarter(fmli_files[-1])),
        "quarters": sorted({_quarter_label(q) for q in df["quarter"].unique()}, key=lambda s: (s.split()[0], s.split()[1][1])),
        "n_files": len(fmli_files),
        "values_capped": n_capped,
        "income_notna": int(df_clean[TARGET].notna().sum()),
    }
    clean = df_clean[df_clean[TARGET].notna()].copy()
    clean = clean.dropna(subset=CONT_FEATS).reset_index(drop=True)
    return clean, report


def build_matrix(df):
    """Build (X_cont raw, X_enc, y) plus a RobustScaler fitted on the data."""
    X_cont = df[CONT_FEATS].copy()
    y = df[TARGET].copy()

    region_dummies = pd.get_dummies(df["REGION"], prefix="region", drop_first=True)
    region_cols = [c for c in ENCODED_FEATS if c.startswith("region_") and c in region_dummies.columns]
    region_dummies = region_dummies[region_cols].astype(float)

    sex_bin = pd.Series((df["SEX_REF"] == 1).astype(int), index=df.index, name="sex_male")
    edu_ord = df["HIGH_EDU"].astype(float).rename("edu_ordinal")

    X_enc = pd.concat([region_dummies.reset_index(drop=True),
                       sex_bin.reset_index(drop=True),
                       edu_ord.reset_index(drop=True)], axis=1)
    X_enc = X_enc.astype(float)

    scaler = RobustScaler()
    X_cont_scaled = scaler.fit_transform(X_cont)
    return X_cont, X_cont_scaled, X_enc.values, y, scaler


def train_model(df, n_estimators=600, verbose=False):
    """Train the transparent Gradient Boosting model; returns (model, scaler, metrics)."""
    X_cont, X_cont_scaled, X_enc, y, scaler = build_matrix(df)
    X_full = np.hstack([X_cont_scaled, X_enc])

    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.2, random_state=42)
    gb = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.04,
        subsample=0.85, max_features=0.8, random_state=42,
    )
    gb.fit(X_tr, y_tr)
    y_pred = gb.predict(X_te)

    metrics = {
        "full_holdout": {
            "r2": float(r2_score(y_te, y_pred)),
            "mae": float(mean_absolute_error(y_te, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
            "n_test": int(len(y_te)),
        }
    }

    # ---- robust estimate on the clean majority (Cook's distance) ----
    try:
        import statsmodels.api as sm
        from statsmodels.tools.tools import add_constant
        X_const = add_constant(X_full)
        cooks = sm.OLS(y, X_const).fit().get_influence().cooks_distance[0]
        clean = cooks <= 4 / len(y)
        Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
            X_full[clean], y[clean], test_size=0.2, random_state=42)
        gb_robust = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=6, learning_rate=0.04,
            subsample=0.85, max_features=0.8, random_state=42)
        gb_robust.fit(Xc_tr, yc_tr)
        y_pred_c = gb_robust.predict(Xc_te)
        metrics["robust"] = {
            "r2": float(r2_score(yc_te, y_pred_c)),
            "mae": float(mean_absolute_error(yc_te, y_pred_c)),
            "rmse": float(np.sqrt(mean_squared_error(yc_te, y_pred_c))),
            "n_clean": int(clean.sum()),
            "n_test": int(len(yc_te)),
        }
        gb = gb_robust
    except Exception as e:  # pragma: no cover - statsmodels optional at runtime
        if verbose:
            print("Robust estimate skipped:", e)

    population = {
        "n": int(len(df)),
        "median": float(y.median()),
        "mean": float(y.mean()),
        "q25": float(y.quantile(0.25)),
        "q75": float(y.quantile(0.75)),
    }

    return gb, scaler, metrics, population


def load_builtin_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base, "output", "features_v1.csv"))
    if TARGET not in df.columns:
        df[TARGET] = df["FINCBTXM"] / 12
    return df[df[TARGET].notna()].copy()


def load_builtin_model():
    """Load the pre-trained honest model artifact; fall back to training on the fly."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "output", "honest_gb.joblib")
    if os.path.exists(path):
        import joblib
        art = joblib.load(path)
        return art["model"], art["scaler"], art["metrics"], art["population"], "pre-trained"
    return None, None, None, None, "not-found"


def save_builtin_model(model, scaler, metrics, population, path=None):
    import joblib
    if path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "output", "honest_gb.joblib")
    joblib.dump(
        {"model": model, "scaler": scaler, "metrics": metrics,
         "population": population, "version": 2},
        path,
    )
    return path