import numpy as np
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

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

EDU_ORDER = [10, 11, 12, 13, 14, 15, 16, 0]

EDU_LABELS = {
    10: "9th-12th no diploma",
    11: "HS no diploma",
    12: "HS graduate",
    13: "Some college",
    14: "Associate degree",
    15: "Bachelor degree",
    16: "Advanced degree",
    0:  "Never attended",
}

REGION_LABELS = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}

MEDIAN_DEFAULTS = {
    "TOTEXPPQ": 9185.0,
    "HOUSPQ": 2988.0,
    "SHELTPQ": 1766.0,
    "TRANSPQ": 846.0,
    "HEALTHPQ": 588.7,
    "EDUCAPQ": 0.0,
    "ENTERTPQ": 200.0,
    "APPARPQ": 130.0,
    "ALCBEVPQ": 75.0,
    "TOBACCPQ": 0.0,
    "food_home": 981.0,
    "food_away": 433.3,
}


def load_model():
    path = os.path.join(OUTPUT_DIR, "final_model.joblib")
    art = joblib.load(path)
    return art["model"]


def fit_scaler_from_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "features_v1.csv"))
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaler.fit(df[CONT_FEATS])
    return scaler


def compute_features(
    age, sex, region, fam_size, perslt18, no_earnr, edu_code,
    totexp_pq, hous_pq, shelter_pq, trans_pq, health_pq,
    educa_pq, enter_pq, app_pq, alc_pq, toba_pq,
    food_home_pq, food_away_pq,
):
    food_total_pq = food_home_pq + food_away_pq

    tot_m = totexp_pq / 3
    if tot_m == 0:
        tot_m = 1e-6

    food_total_m = food_total_pq / 3
    food_home_m = food_home_pq / 3
    food_away_m = food_away_pq / 3
    hous_m = hous_pq / 3
    shelter_m = shelter_pq / 3
    trans_m = trans_pq / 3
    health_m = health_pq / 3
    educa_m = educa_pq / 3
    enter_m = enter_pq / 3
    app_m = app_pq / 3
    alc_m = alc_pq / 3
    toba_m = toba_pq / 3

    def safe_share(cat_pq):
        return (cat_pq / 3) / tot_m

    share_food = safe_share(food_total_pq)
    share_food_home = safe_share(food_home_pq)
    share_food_away = safe_share(food_away_pq)
    share_housing = safe_share(hous_pq)
    share_shelter = safe_share(shelter_pq)
    share_transport = safe_share(trans_pq)
    share_health = safe_share(health_pq)
    share_education = safe_share(educa_pq)
    share_entertainment = safe_share(enter_pq)
    share_apparel = safe_share(app_pq)
    share_alcohol = safe_share(alc_pq)
    share_tobacco = safe_share(toba_pq)

    top_level = (share_food + share_housing + share_transport + share_health
                 + share_education + share_entertainment + share_apparel
                 + share_alcohol + share_tobacco)
    share_others = max(0, 1 - top_level)

    adults = max(fam_size - perslt18, 1)
    children = max(perslt18, 0)
    oecd_scale = 1 + 0.7 * (adults - 1) + 0.5 * children
    percapita_exp_m = tot_m / oecd_scale

    sex_male = 1 if sex == 1 else 0
    region_2 = 1 if region == 2 else 0
    region_3 = 1 if region == 3 else 0
    region_4 = 1 if region == 4 else 0
    edu_ordinal = float(edu_code)

    row_cont = np.array([[
        tot_m, food_total_m, food_home_m, food_away_m,
        hous_m, shelter_m, trans_m, health_m, educa_m,
        enter_m, app_m, alc_m, toba_m,
        age, fam_size, perslt18, no_earnr,
        share_food, share_food_home, share_food_away, share_housing,
        share_shelter, share_transport, share_health, share_education,
        share_entertainment, share_apparel, share_alcohol, share_tobacco,
        share_others, oecd_scale, percapita_exp_m,
    ]])

    row_enc = np.array([[region_2, region_3, region_4, sex_male, edu_ordinal]])

    return row_cont, row_enc


def predict(row_cont, row_enc, model, scaler):
    row_scaled = scaler.transform(row_cont)
    X_full = np.hstack([row_scaled, row_enc])
    prediction = model.predict(X_full)[0]
    return max(0, prediction)
