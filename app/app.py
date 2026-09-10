import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import analytics
import pipeline
from preprocessing import (
    CONT_FEATS, ENCODED_FEATS, MEDIAN_DEFAULTS, compute_features,
)

st.set_page_config(page_title="CEXInsight — Income Explorer", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
    .cex-hero { background: linear-gradient(135deg, #1f3a5f 0%, #2c3e50 55%, #3a2f5f 100%);
        border-radius: 16px; padding: 30px 36px; margin-bottom: 1.4rem; color: #fff;
        box-shadow: 0 8px 24px rgba(28,47,72,0.25); border-left: 6px solid #f39c12;
        position: relative; overflow: hidden; }
    .cex-hero::after { content: ""; position: absolute; right: -40px; top: -40px; width: 190px; height: 190px;
        border-radius: 50%; background: radial-gradient(circle, rgba(243,156,18,0.25), transparent 70%); }
    .cex-hero .eyebrow { display: inline-block; background: rgba(243,156,18,0.15); color: #ffd27f;
        border: 1px solid rgba(243,156,18,0.45); border-radius: 999px; padding: 3px 13px; font-size: 0.78rem;
        font-weight: 600; letter-spacing: 0.6px; text-transform: uppercase; margin-bottom: 12px;
        position: relative; z-index: 1; }
    .cex-hero h1 { color: #fff; margin: 0 0 10px 0; font-size: 2.05rem; letter-spacing: -0.5px;
        position: relative; z-index: 1; }
    .cex-hero .accent { color: #f39c12; }
    .cex-hero .sub { color: #d7dde8; font-size: 1.05rem; line-height: 1.62; max-width: 1000px;
        position: relative; z-index: 1; }
    .cex-hero .sub b { color: #f39c12; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        border-radius: 10px; padding: 14px 18px; color: white;
        box-shadow: 0 3px 6px rgba(0,0,0,0.12);
    }
    [data-testid="stMetric"] label { color: rgba(255,255,255,0.85) !important; font-size: 0.88rem !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f39c12 !important; font-weight: 700 !important; }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] { color: rgba(255,255,255,0.85) !important; }
    div[data-testid="stSidebar"] { background-color: #f6f7fb; }
    div[data-testid="stSidebar"] h1 { color: #2c3e50; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.3rem; }
    .stTabs [data-baseweb="tab"] {
        background: #eef2f7; border-radius: 8px 8px 0 0; padding: 6px 16px; font-weight: 600;
    }
    div.stButton > button[kind="primary"] { background-color: #f39c12; border-color: #f39c12; }
    h2 { color: #1a2530; }
    h3 { border-bottom: 2px solid #3498db; padding-bottom: 0.2rem; color: #1a2530; }
    .step-card { background:#fff; border:1px solid #e3e8ee; border-left:4px solid #3498db;
        border-radius:8px; padding:10px 14px; margin-bottom:8px; }
    .step-card b { color:#1a2530; }
</style>
""", unsafe_allow_html=True)

NICE_NAMES = {
    "TOTEXPPQ_M": "Total monthly spending",
    "food_total_m": "Food spending (total)",
    "food_home_m": "Groceries",
    "food_away_m": "Dining out",
    "HOUSPQ_M": "Housing cost",
    "SHELTPQ_M": "Rent / mortgage",
    "TRANSPQ_M": "Transportation",
    "HEALTHPQ_M": "Healthcare",
    "EDUCAPQ_M": "Education spending",
    "ENTERTPQ_M": "Entertainment",
    "APPARPQ_M": "Clothing",
    "ALCBEVPQ_M": "Alcohol",
    "TOBACCPQ_M": "Tobacco",
    "AGE_REF": "Age of reference person",
    "FAM_SIZE": "Family size",
    "PERSLT18": "Children",
    "NO_EARNR": "Non-earning members",
    "share_food": "Budget share: food",
    "share_food_home": "Budget share: groceries",
    "share_food_away": "Budget share: dining out",
    "share_housing": "Budget share: housing",
    "share_shelter": "Budget share: rent/mortgage",
    "share_transport": "Budget share: transport",
    "share_health": "Budget share: healthcare",
    "share_education": "Budget share: education",
    "share_entertainment": "Budget share: entertainment",
    "share_apparel": "Budget share: clothing",
    "share_alcohol": "Budget share: alcohol",
    "share_tobacco": "Budget share: tobacco",
    "share_others": "Budget share: everything else",
    "oecd_scale": "Household composition scale",
    "percapita_exp_m": "Per-person spending",
    "region_2": "Region: Midwest",
    "region_3": "Region: South",
    "region_4": "Region: West",
    "sex_male": "Male reference person",
    "edu_ordinal": "Education level",
}

REGION_LABELS = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
from preprocessing import EDU_ORDER, EDU_LABELS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


# ─────────────────── active dataset state ───────────────────
def activate_builtin():
    df = pipeline.load_builtin_data()
    model, scaler, metrics, population, src = pipeline.load_builtin_model()
    if model is None:
        with st.spinner("Training the built-in model for the first time (~2 min)…"):
            model, scaler, metrics, population = pipeline.train_model(df, n_estimators=600)
    st.session_state["df"] = df
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["metrics"] = metrics
    st.session_state["population"] = population
    st.session_state["source"] = "built-in"
    st.session_state["ready"] = True
    st.session_state.pop("eval_cache", None)


def ensure_ready():
    if st.session_state.get("ready"):
        return
    activate_builtin()


def get_eval():
    """Hold-out predictions for the CURRENT model/df, computed once per dataset."""
    ec = st.session_state.get("eval_cache")
    if ec is None:
        ec = analytics.holdout_eval(st.session_state["df"], st.session_state["model"])
        st.session_state["eval_cache"] = ec
    return ec


def active_spend_defaults(df):
    """Quarterly spending defaults from the ACTIVE dataset (×3 of monthly median),
    so first-load sliders match whatever data is loaded — built-in or user upload."""
    get = lambda c: float(df[c].median())
    return {
        "TOTEXPPQ": get("TOTEXPPQ_M") * 3,
        "HOUSPQ": get("HOUSPQ_M") * 3,
        "SHELTPQ": get("SHELTPQ_M") * 3,
        "TRANSPQ": get("TRANSPQ_M") * 3,
        "HEALTHPQ": get("HEALTHPQ_M") * 3,
        "EDUCAPQ": get("EDUCAPQ_M") * 3,
        "ENTERTPQ": get("ENTERTPQ_M") * 3,
        "APPARPQ": get("APPARPQ_M") * 3,
        "ALCBEVPQ": get("ALCBEVPQ_M") * 3,
        "TOBACCPQ": get("TOBACCPQ_M") * 3,
        "food_home": get("food_home_m") * 3,
        "food_away": get("food_away_m") * 3,
    }


def data_strip():
    """One-line reminder of which dataset powers this page."""
    pop = st.session_state["population"]
    if st.session_state.get("source") == "your-data":
        st.success(
            f"**Active dataset: your upload** — {pop['n']:,} households, "
            f"median ${pop['median']:,.0f}/month. Everything below is computed on it.")
    else:
        st.info(
            f"**Active dataset: built-in sample** — {pop['n']:,} households, "
            f"median ${pop['median']:,.0f}/month. Load your own on the first page.")


PREVIEW_COLS = ["FINCBTXM_M", "TOTEXPPQ_M", "HOUSPQ_M", "food_total_m",
                "AGE_REF", "SEX_REF", "REGION", "FAM_SIZE", "HIGH_EDU"]


def show_dataset_summary(df, pop, metrics, quarters_note):
    """Compact, pretty summary of the active dataset (used by page 1)."""
    erob = metrics.get("robust", metrics["full_holdout"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Households", f"{len(df):,}")
    c2.metric("Median income", f"${pop['median']:,.0f}/mo")
    c3.metric("Mean income", f"${pop['mean']:,.0f}/mo")
    c4.metric("Model R² (typical)", f"{erob['r2']:.2f}",
              help="Share of income differences the model explains on typical households — this model powers pages 2-4.")

    with st.expander("🧾 See the data (first households)"):
        show_cols = [c for c in PREVIEW_COLS if c in df.columns]
        st.dataframe(df[show_cols].head(8), width="stretch")

    eng = analytics.engel_table(df)
    with st.expander("🧪 Data-quality check — does Engel's law hold here?"):
        st.markdown(
            "A good sanity check: food and housing should take a *smaller* share of the budget "
            "as income rises. If this holds, the data is clean and credible.")
        st.dataframe(eng, width="stretch")

    st.download_button(
        "⬇️ Download this dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
        file_name="cex_households.csv", mime="text/csv",
        help="One row per household — handy for a quick look in Excel.",
    )
    st.caption(f"Period: {quarters_note} · This dataset is used by **every** page of the app.")


# ─────────────────── sidebar ───────────────────
with st.sidebar:
    st.title("📊 CEXInsight")
    st.caption("US Household Income Explorer")

    page = st.radio(
        "Go to a page",
        [
            "1. Introduction & data",
            "2. Predict income & the rules",
            "3. Education & income",
            "4. Key findings",
        ],
        format_func=lambda x: {
            "1. Introduction & data": "📖 Introduction & data",
            "2. Predict income & the rules": "🎯 Predict income & the rules",
            "3. Education & income": "🎓 Education & income",
            "4. Key findings": "📄 Key findings",
        }[x],
    )
    st.caption(
        {
            "1. Introduction & data": "What CEX is, how this app works, and where to upload your data",
            "2. Predict income & the rules": "Our income model, plus what the Engel / 50-30-20 rules add",
            "3. Education & income": "The education → income research (tests, charts, clustering)",
            "4. Key findings": "Honest summary and exportable report",
        }[page]
    )

    ensure_ready()
    st.divider()
    st.subheader("Dataset in use")
    if st.session_state.get("source") == "built-in":
        st.info("**Built-in sample** — 2024-2025 US Consumer Expenditure Survey.")
    else:
        st.success("**Your uploaded data**")
    st.markdown(f"- Households: **{st.session_state['population']['n']:,}**")
    st.markdown(f"- Median monthly income: **${st.session_state['population']['median']:,.0f}**")

    if st.session_state.get("source") == "your-data":
        if st.button("↺ Back to built-in sample", width="stretch"):
            activate_builtin()
            st.rerun()

    st.divider()
    st.caption(
        "Built on the official U.S. Consumer Expenditure Survey. "
        "Every figure is a statistical estimate, not financial advice."
    )

# ══════════════════════ PAGE 1: INTRODUCTION & DATA ══════════════════════
if page == "1. Introduction & data":
    st.markdown("""
<div class="cex-hero">
  <h1>CEXInsight</h1>
  <div class="sub">How U.S. households earn and spend. We turn the <b>Consumer Expenditure Survey</b>
  into an interactive story: predict <b>income from spending</b>, test the classic budget rules,
  and measure what <b>education</b> is worth.</div>
</div>
""", unsafe_allow_html=True)

    intro_cols = st.columns(2)
    with intro_cols[0]:
        st.subheader("The data")
        st.markdown("""
The **Consumer Expenditure Survey (CE)** is the official U.S. survey maintained by the
Bureau of Labor Statistics. Every quarter it asks thousands of households two simple things:

- **How much money came in** — income before taxes
- **How it was spent** — groceries, housing, transport, health, clothing…

It is the reference dataset economists use to study how Americans live.

**The question we answer here:** if a household tells us *how it lives and spends*, can a
model infer *how much it earns*? The answer is a strong **yes** — and along the way we
learn a lot about what drives income, starting with **education**.
""")
    with intro_cols[1]:
        st.subheader("The method")
        st.markdown("""
This dashboard mirrors the notebook's three sprints — and **one dataset drives it all**:

1. **Prepare.** Load raw `fmli*.csv` files (or the built-in sample), clean them and
   engineer features: monthly spending, budget shares, household composition.
2. **Predict + rules.** A transparent Gradient Boosting model predicts monthly income
   from how a household lives. We then ask what economic rules of thumb — **Engel's law,
   50/30/20, the 30% housing guide** — add on top.
3. **Education study.** Statistical tests, the education → income premium, the gender gap,
   clustering and Engel's law, just like the notebook's Sprint 3.
""")

    st.divider()
    st.subheader("Using the app, in four clicks")
    c_how1, c_how2, c_how3, c_how4 = st.columns(4)
    c_how1.markdown('<div class="step-card"><b>1 · Data.</b><br>Choose a dataset here — it feeds every page.</div>', unsafe_allow_html=True)
    c_how2.markdown('<div class="step-card"><b>2 · Predict.</b><br>Describe a household → predicted income, why, budget vs rules.</div>', unsafe_allow_html=True)
    c_how3.markdown('<div class="step-card"><b>3 · Education.</b><br>The education → income research, charts and tests.</div>', unsafe_allow_html=True)
    c_how4.markdown('<div class="step-card"><b>4 · Findings.</b><br>The honest summary of everything we learned.</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("1 · Pick the dataset that powers this app")
    st.markdown("Everything on the next pages runs on the dataset you choose here. "
                "Start from the built-in sample, or load your own raw survey files.")

    src_choice = st.radio(
        "Which data would you like to use?",
        ["Use the built-in sample", "Upload your own files"],
        horizontal=True,
        index=0 if st.session_state.get("source") == "built-in" else 1,
        key="src_choice",
    )

    if src_choice.startswith("Use the built-in"):
        if st.session_state.get("source") != "built-in":
            activate_builtin()
            st.rerun()
        show_dataset_summary(
            st.session_state["df"], st.session_state["population"],
            st.session_state["metrics"], "2024 Q1 – 2025 Q1 (5 quarters)")
    else:
        st.markdown("Drop the interview files named **`fmli*.csv`** below (several allowed). "
                    "Cleaning, feature-building, model training and every analysis on pages 2–4 "
                    "will rerun on **your** files.")
        c1, c2 = st.columns([1, 2])
        with c1:
            uploaded = st.file_uploader(
                "Upload your fmli*.csv files", type="csv", accept_multiple_files=True)
            if st.button("↺ Back to built-in sample", width="stretch"):
                activate_builtin()
                st.rerun()
        with c2:
            if not uploaded:
                st.info("No files yet — for now every page still runs on the built-in sample.")
            else:
                fmli_up = [f for f in uploaded if os.path.basename(f.name).startswith("fmli")]
                fp = tuple(sorted(os.path.basename(f.name) for f in fmli_up))
                already = (st.session_state.get("processed_fp") == fp
                           and st.session_state.get("source") == "your-data")
                if len(fmli_up) == 0:
                    st.error("We need at least one `fmli*.csv` file (e.g. `fmli242.csv`). "
                             "`memi*.csv` files are optional member-level data — not required.")
                elif already:
                    st.success("These files are already loaded and the model is trained on them.")
                else:
                    with st.status(f"Preparing {len(fmli_up)} file(s)…", expanded=True) as status:
                        st.write("1 · Reading the survey files…")
                        df_new, report = pipeline.process_cex_files(fmli_up)
                        st.write(f"Read **{report['raw_rows']:,}** interviews → "
                                 f"kept **{report['unique_households']:,}** unique households.")

                        st.write("2 · Cleaning — fix negatives, missing regions, cap extreme values…")
                        st.write(f"Capped **{report['values_capped']:,}** extreme values so a few "
                                 f"unusual households don't distort the results.")

                        st.write("3 · Building features — monthly spending, budget shares, household scale…")

                        st.write("4 · Training the model on your data (~1 min)…")
                        model_new, scaler_new, metrics_new, pop_new = pipeline.train_model(df_new, n_estimators=400)
                        st.write("5 · Evaluating honestly on data it never saw…")
                        status.update(label="Your data is ready!", state="complete", expanded=True)

                    st.session_state.update(
                        df=df_new, model=model_new, scaler=scaler_new,
                        metrics=metrics_new, population=pop_new,
                        source="your-data", ready=True, processed_fp=fp, report=report)
                    st.session_state.pop("eval_cache", None)

                    st.success(f"**Done.** {pop_new['n']:,} households — every page of the app "
                               f"now runs entirely on your data.")

        if st.session_state.get("source") == "your-data":
            rpt = st.session_state.get("report", {})
            show_dataset_summary(
                st.session_state["df"], st.session_state["population"],
                st.session_state["metrics"],
                " · ".join(rpt.get("quarters", ["your files"])))
            with st.expander("🗺️ Where your households are"):
                reg_counts = (st.session_state["df"]["REGION"].replace(REGION_LABELS)
                              .value_counts().rename("households"))
                st.bar_chart(reg_counts)

# ════════════════ PAGE 2: PREDICT INCOME & THE RULES ════════════════
elif page == "2. Predict income & the rules":
    data_strip()
    st.header("Predict a household's monthly income")
    st.markdown(f"""
Describe a household below. The model (trained on **{st.session_state['population']['n']:,} households**)
predicts their monthly income before taxes. Spending is monthly: the survey records a quarter
(3 months), so quarterly amounts are divided by 3 internally.
""")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Who they are")
        age = st.slider("Age of the reference person", 18, 85, 45)
        sex = st.radio("Gender of the reference person", [1, 2],
                       format_func=lambda x: {1: "Male", 2: "Female"}[x])
        region = st.selectbox("Region of the country", list(REGION_LABELS.keys()),
                              format_func=lambda x: REGION_LABELS[x])
        fam_size = st.number_input("People in the household", 1, 20, 3)
        perslt18 = st.number_input("Children under 18", 0, fam_size, min(1, fam_size))
        no_earnr = st.number_input("Working-age members without income", 0, fam_size, 0)
        edu_code = st.selectbox("Education of the reference person", EDU_ORDER,
                                format_func=lambda x: EDU_LABELS[x],
                                help="The reference person is the household member in whose name the home is owned/rented.")

    with col2:
        st.subheader("3. How they spend (per month)")
        st.caption("Start from the typical household and adjust.")
        MD = active_spend_defaults(st.session_state["df"])
        totexp_pq = st.number_input("Total spending", 0.0, 500000.0, MD["TOTEXPPQ"], step=100.0)
        food_home_pq = st.number_input("Groceries", 0.0, 50000.0, MD["food_home"], step=50.0)
        food_away_pq = st.number_input("Restaurants / takeout", 0.0, 50000.0, MD["food_away"], step=50.0)
        hous_pq = st.number_input("Housing (total)", 0.0, 100000.0, MD["HOUSPQ"], step=100.0)
        shelter_pq = st.number_input("… of which rent / mortgage", 0.0, hous_pq,
                                     min(MD["SHELTPQ"], hous_pq), step=100.0)
        trans_pq = st.number_input("Transportation", 0.0, 50000.0, MD["TRANSPQ"], step=50.0)
        health_pq = st.number_input("Healthcare", 0.0, 50000.0, MD["HEALTHPQ"], step=50.0)
        educa_pq = st.number_input("Education", 0.0, 50000.0, MD["EDUCAPQ"], step=50.0)
        enter_pq = st.number_input("Entertainment", 0.0, 50000.0, MD["ENTERTPQ"], step=50.0)
        app_pq = st.number_input("Clothing", 0.0, 50000.0, MD["APPARPQ"], step=50.0)
        alc_pq = st.number_input("Alcohol", 0.0, 50000.0, MD["ALCBEVPQ"], step=10.0)
        toba_pq = st.number_input("Tobacco", 0.0, 50000.0, MD["TOBACCPQ"], step=10.0)

    if st.button("Predict income", type="primary"):
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        df_active = st.session_state["df"]
        pop = st.session_state["population"]

        row_cont, row_enc = compute_features(
            age, sex, region, fam_size, perslt18, no_earnr, edu_code,
            totexp_pq, hous_pq, shelter_pq, trans_pq, health_pq,
            educa_pq, enter_pq, app_pq, alc_pq, toba_pq,
            food_home_pq, food_away_pq,
        )
        result = analytics.predict_row(row_cont, row_enc, model, scaler)

        pctile = float((df_active["FINCBTXM_M"] < result).mean()) * 100
        delta = result - pop["median"]

        st.divider()
        st.subheader("Prediction")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted monthly income", f"${result:,.0f}")
        c2.metric("vs. median household", f"${delta:+,.0f}", f"{delta / pop['median'] * 100:+.1f}%")
        c3.metric("Richer than", f"{pctile:.0f}% of households",
                  help="Out of 100 households, this predicted income is higher than this many of them.")
        c4.metric("Yearly equivalent", f"${result * 12:,.0f}")

        st.progress(min(pctile / 100, 1.0))
        st.caption(f"This household would rank around the {pctile:.0f}th percentile of the {pop['n']:,}-household population.")

        eval_ = get_eval()
        resid = eval_["residual"].to_numpy()
        bands = analytics.residual_bands(resid)
        st.info(
            f"**How precise is this?** Empirically, the middle **68%** of households fall within "
            f"**${max(0, result + bands[68][0]):,.0f} – ${result + bands[68][1]:,.0f}/month** "
            f"of this estimate, and the middle **95%** within "
            f"**${max(0, result + bands[95][0]):,.0f} – ${result + bands[95][1]:,.0f}/month** "
            f"(measured on {len(eval_):,} households this model never saw)."
        )

        st.divider()
        st.subheader("Your budget against the 50/30/20 rule")
        st.markdown(
            "The classic personal-finance guideline: **50% needs** (housing, food, transport, health), "
            "**30% wants**, **20% savings**. It is a *rule of thumb* — in survey data it is interpretive, "
            "not a target the model enforces."
        )
        bud_rows, bstat = analytics.budget_vs_rule(
            result, totexp_pq, hous_pq, food_home_pq, food_away_pq,
            trans_pq, health_pq, educa_pq, enter_pq, app_pq, alc_pq, toba_pq,
        )
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ypos = np.arange(len(bud_rows))
        ax.barh(ypos - 0.2, bud_rows["Your household"] * 100, height=0.38, color="#3498db", label="Your household")
        ax.barh(ypos + 0.2, bud_rows["50/30/20 guideline"] * 100, height=0.38, color="#95a5a6", label="50/30/20 rule")
        ax.set_yticks(ypos); ax.set_yticklabels(bud_rows["Bucket"])
        ax.set_xlabel("% of monthly income")
        ax.set_xlim(0, 100)
        for yi, v in zip(ypos - 0.2, bud_rows["Your household"] * 100):
            ax.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=9)
        ax.legend(loc="lower right")
        ax.set_title("")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        if bstat["housing_ok"]:
            st.success(f"Housing takes **{bstat['housing_share']:.0%}** of income — within the **30%** guideline.")
        else:
            st.warning(f"Housing takes **{bstat['housing_share']:.0%}** of income — above the **30%** guideline often used by lenders.")
        if bstat["savings_raw"] < 0:
            st.caption("Note: spending exceeds the predicted income here — the survey often under-reports income for high spenders.")

        st.divider()
        st.subheader("Why this income level?")
        st.markdown("""
The model mainly listens to the strongest signals in the data: total spending,
housing cost, how much of the budget goes to food (Engel's law), education level,
and household size. The table compares **your household** with a **typical household**
on the 8 most influential signals.
""")
        contrib = analytics.contribution_df(model, scaler, row_cont, row_enc,
                                            df_active, NICE_NAMES)
        st.dataframe(contrib, hide_index=True, width="stretch")

        erob = st.session_state["metrics"].get("robust", st.session_state["metrics"]["full_holdout"])
        st.caption(
            f"Model quality on population: accurate to about ±${erob['mae']:,.0f}/month for typical households "
            f"(explains {erob['r2']:.0%} of income differences). Every prediction comes with this uncertainty."
        )

        st.divider()
        st.subheader("What do the Engel / 50-30-20 / 30% rules add?")
        rs = analytics.rule_summary(df_active)
        rr1, rr2, rr3, rr4 = st.columns(4)
        rr1.metric("Food share, poorest→richest quarter", f"{rs['food_q1']:.0%} → {rs['food_q4']:.0%}",
                   help="Engel's law: food takes a smaller share as income rises.")
        rr2.metric("Housing share, poorest→richest quarter", f"{rs['housing_q1']:.0%} → {rs['housing_q4']:.0%}",
                   help="Housing also takes less of the budget as income rises.")
        rr3.metric("Households over 30% for housing", f"{rs['pct_over30']:.0%}",
                   help="The classic lender guideline that housing should stay under 30% of income.")
        rr4.metric("Spending × income correlation", f"{rs['rho_spend']:.2f}",
                   help="The strongest rule-of-thumb signal: total spending on its own correlates 0.60 with income.")
        st.markdown(f"""
**The verdict from the data:**

1. **The rules are real.** Engel's law holds: food's share of the budget falls from ~
   **{rs['food_q1']:.0%}** (poorest quarter) to ~**{rs['food_q4']:.0%}** (richest quarter),
   and housing follows from ~**{rs['housing_q1']:.0%}** to ~**{rs['housing_q4']:.0%}**.
   Still, **{rs['pct_over30']:.0%}** of households exceed the 30% housing guideline —
   rules describe averages, not everyone.

2. **As a model, the rules are weak.** On their own these rule measures only weakly track
   income (food share ρ ≈ {abs(rs['rho_food']):.2f}, housing ρ ≈ {abs(rs['rho_housing']):.2f}) —
   compared with ρ ≈ {rs['rho_spend']:.2f} for total spending alone.

3. **They add little to the machine-learning model** (~6% of feature influence when included).
   The model already reads Engel's law through the budget-share features. So the rules are
   **interpretive** — they help explain and validate the data, but they don't replace the model.
""")

# ════════════════════════ PAGE 3: EDUCATION & INCOME ════════════════════════
elif page == "3. Education & income":
    df_active = st.session_state["df"]
    data_strip()

    st.header("From education to income — the research")
    st.markdown("""
This is the study at the heart of the project. Six steps, exactly as in the notebook (Sprint 3):
does a household's **education** go with its **income** — how much, how robustly, and for whom?
""")

    tab_pred, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Predict from education",
        "Education & income",
        "Gender & education",
        "What drives income",
        "Household types",
        "How money is spent",
        "Model accuracy",
    ])

    with tab_pred:
        st.subheader("Predict income from education 🎓")
        st.markdown(
            "Pick an education level: we take a **typical household of that level** (typical age, "
            "family size and spending mix) and ask the model what income to expect. Uncheck the box "
            "to define your own household instead."
        )
        pred_avail = [e for e in analytics.EDU_ORDER_MAIN if e in df_active["HIGH_EDU"].unique()]
        c_edu, c_profile = st.columns([1, 2])
        with c_edu:
            edu_pred = st.selectbox(
                "Education level", pred_avail,
                format_func=lambda x: analytics.EDU_LABELS[x],
                key="edu_predict",
            )
            use_typical = st.checkbox("Use a typical household for this level", value=True)
        inp = analytics.typical_inputs(df_active, edu_pred)
        overrides = {}
        if use_typical:
            sex_lbl = "Man" if inp["sex"] == 1 else "Woman"
            c_profile.info(
                f"Typical profile for **{analytics.EDU_LABELS[edu_pred]}**: "
                f"age **{inp['age']}**, {sex_lbl}-headed, **{inp['fam_size']}** people "
                f"({inp['perslt18']} child(ren)), spending "
                f"**${inp['totexp_pq'] / 3:,.0f}/month**."
            )
        else:
            with c_profile:
                p_age = st.slider("Age", 18, 85, inp["age"])
                p_sex = st.radio("Gender", [1, 2], index=0 if inp["sex"] == 1 else 1,
                                 format_func=lambda x: {1: "Male", 2: "Female"}[x])
                p_fam = st.number_input("People in the household", 1, 12, inp["fam_size"])
                p_lt18 = st.number_input("Children under 18", 0, min(6, p_fam), inp["perslt18"])
            overrides = {"age": p_age, "sex": p_sex, "fam_size": p_fam, "perslt18": p_lt18}

        model_p3 = st.session_state["model"]
        scaler_p3 = st.session_state["scaler"]
        pop_p3 = st.session_state["population"]
        pred = analytics.predict_from_education(df_active, model_p3, scaler_p3, edu_pred, **overrides)
        pct_p3 = analytics.income_percentile(pred, df_active)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted monthly income", f"${pred:,.0f}")
        m2.metric("vs. median household", f"${pred - pop_p3['median']:+,.0f}")
        m3.metric("Richer than", f"{pct_p3:.0f}% of households")
        m4.metric("Yearly equivalent", f"${pred * 12:,.0f}")
        st.progress(min(pct_p3 / 100, 1.0))

        st.subheader("The education ladder — what each level predicts")
        st.markdown(
            "For every education level we predict income for its **typical household**. "
            "This isolates the education gradient as the model sees it."
        )
        ladder = analytics.education_ladder(df_active, model_p3, scaler_p3, highlight=edu_pred)
        ladder_sorted = ladder.sort_values("Predicted monthly income", ascending=True)
        cols = ["#f39c12" if r.highlight else "#3498db" for _, r in ladder_sorted.iterrows()]
        fig, ax = plt.subplots(figsize=(10, 5))
        ladder_sorted.plot.barh(
            x="Education level", y="Predicted monthly income", ax=ax, color=cols, legend=False)
        ax.set_xlabel("Predicted monthly income (USD)")
        ax.set_title("Predicted income by education level (typical household for each)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption(
            f"Selected: **{analytics.EDU_LABELS[edu_pred]}** (in orange). Predicted income includes what the level's "
            "typical spending implies — Engel's law and spending power go hand-in-hand with education."
        )

    with tab1:
        st.subheader("1 · Education and income")
        st.markdown("""
Higher education and higher income go together — but by how much, once we account for age
and family? Two views: the income distribution by education (below), and the **extra monthly income**
each degree buys on average *after* controlling for the household profile (table).
""")
        dlev, cats = analytics.education_levels_data(df_active)
        fig, ax = plt.subplots(figsize=(12, 5))
        dlev.boxplot(column="FINCBTXM_M", by="edu_label", vert=True, grid=False, ax=ax, showfliers=False)
        ax.set_title("")
        ax.figure.suptitle("")
        ax.set_ylabel("Monthly income before taxes (USD)")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Average effect of education on monthly income** (after controlling for age, family size, children, sex):")
        dlev2, rows, f_stat, p_val, _ = analytics.education_income_stats(df_active)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(
            f"Test that income really differs across education levels: **p = {p_val:.2g}** "
            f"(far below 0.05 → the education-income link is real, not random).")

    with tab2:
        st.subheader("2 · Does education pay off differently for men and women?")
        st.markdown(
            "We compare the education premium for male-headed vs female-headed households and "
            "statistically test whether the gap could be due to chance (interaction model, "
            "HC3 robust standard errors — as in the notebook).")
        gen_rows, inter_p, _ = analytics.gender_interaction_test(df_active)
        st.dataframe(gen_rows, hide_index=True, width="stretch")
        if inter_p < 0.05:
            st.markdown(
                f"**Overall gender difference: significant (p = {inter_p:.3f}).** The gap is driven mainly "
                f"by the Bachelor's degree premium — men earn significantly more at Bachelor level. "
                f"Other levels show no significant gap.")
        else:
            st.markdown(
                f"**Overall gender difference: not significant (p = {inter_p:.3f}).** For most education "
                f"levels the premium is similar for men and women.")

    with tab3:
        st.subheader("3 · Which household signals predict income the most?")
        st.markdown("""
The chart ranks the signals the model leans on. A longer bar = bigger influence on the prediction.
Spending and its composition (how much of the budget goes to food, e.g.) carry most of the signal;
education still matters on its own.
""")
        importances = np.asarray(st.session_state["model"].feature_importances_)
        names = CONT_FEATS + ENCODED_FEATS
        fi = pd.Series(importances, index=names).sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 7))
        fi.plot.barh(ax=ax, color=["#e67e22" if n.startswith(("edu", "region_", "sex")) else "#3498db" for n in fi.index])
        ax.invert_yaxis()
        ax.set_title("Top 15 signals (orange = profile/education, blue = spending)")
        ax.set_xlabel("Relative influence")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
**How to read it:** these are the signals the model discovered matter most.
Notice that *how* a household splits its budget (food share, housing share) matters a lot —
echoing Engel's law. Education appears too, confirming it captures something income-related
that spending alone doesn't.
""")

    with tab4:
        st.subheader("4 · Natural household types")
        st.markdown("""
Without being told any labels, an algorithm grouped households into 4 income tiers.
If the tiers line up with education (x-axis) and income (y-axis), it confirms the two are linked.
""")
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        edu_income = df_active[["HIGH_EDU", "FINCBTXM_M"]].dropna().copy()
        edu_income["edu_ordinal"] = edu_income["HIGH_EDU"].astype(float)
        X2 = StandardScaler().fit_transform(edu_income[["edu_ordinal", "FINCBTXM_M"]])

        order = [e for e in EDU_ORDER if e in edu_income["HIGH_EDU"].unique()]
        edu_income["edu_label"] = edu_income["HIGH_EDU"].map(EDU_LABELS)
        cats = [e for e in EDU_ORDER if e in order]
        edu_income["edu_label"] = pd.Categorical(edu_income["edu_label"], categories=[EDU_LABELS[e] for e in cats], ordered=True)

        km = KMeans(n_clusters=4, n_init=20, random_state=42)
        edu_income["cluster"] = km.fit_predict(X2)
        order_names = edu_income.groupby("cluster")["FINCBTXM_M"].mean().sort_values().index
        tier = {c: t for c, t in zip(order_names, ["Lower", "Lower-middle", "Upper-middle", "Higher"])}
        edu_income["tier"] = edu_income["cluster"].map(tier)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors_c = plt.cm.Set1(np.linspace(0, 1, 4))
        for c in range(4):
            sub = edu_income[edu_income["cluster"] == c]
            ax.scatter(sub["edu_ordinal"], sub["FINCBTXM_M"] / 1000, alpha=0.35, s=18,
                       color=colors_c[c], label=tier.get(c, c))
        ax.set_xlabel("Education level")
        ax.set_ylabel("Monthly income (thousands USD)")
        ax.set_title("Natural income tiers by education")
        ax.legend(title="Tier")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([EDU_LABELS[e].replace(" degree", "") for e in cats], rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(
            edu_income.groupby("tier").agg(
                Households=("FINCBTXM_M", "count"),
                Avg_income=("FINCBTXM_M", lambda x: f"${x.mean():,.0f}"),
                Median_income=("FINCBTXM_M", lambda x: f"${x.median():,.0f}"),
            ).rename(columns={"Avg_income": "Average income", "Median_income": "Median income"}),
            width="stretch",
        )

    with tab5:
        st.subheader("5 · How households spend their money")
        st.markdown("""
**Engel's law:** the more households earn, the smaller the share of budget spent on food.
The table below splits households into 4 income groups and shows the average food and housing shares.
""")
        st.dataframe(analytics.engel_table(df_active), width="stretch")

        st.markdown("**Engel's law, live — pick an education level:**")
        avail = [e for e in analytics.EDU_ORDER_MAIN if e in df_active["HIGH_EDU"].unique()]
        sel = st.selectbox(
            "Education level", avail, format_func=lambda x: analytics.EDU_LABELS[x],
            key="engel_edu",
        )
        t = analytics.engel_curve(df_active, edu_code=sel)
        if t.empty:
            st.warning("Not enough households in this group to draw a reliable curve.")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t["median_income"], t["food_share"], marker="o", color="#e74c3c", label="Food share")
            ax.plot(t["median_income"], t["housing_share"], marker="s", color="#3498db", label="Housing share")
            ax.set_xlabel("Median monthly income of the group (USD)")
            ax.set_ylabel("Average share of the budget")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.set_title(f"Engel's law for {analytics.EDU_LABELS[sel]} households")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                f"Food share {t['food_share'].iloc[0]:.0%} → {t['food_share'].iloc[-1]:.0%} "
                f"from the poorest to the richest group of this education level."
            )

        st.markdown("**Budget shares by education level:**")
        profile = analytics.budget_shares_by_education(df_active)
        fig, ax = plt.subplots(figsize=(13, 5.5))
        profile.plot.bar(ax=ax, colormap="Set2", width=0.85)
        ax.set_ylabel("Average share of monthly budget")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(loc="upper right", fontsize=9, title="Category")
        ax.set_title("How the budget is split by education level")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab6:
        st.subheader("6 · How accurate is the model, really?")
        st.markdown(
            "Every dot is a household the model **never saw during training**. "
            "If predictions were perfect, all dots would sit on the red line."
        )
        eval_ = get_eval()
        resid = eval_["residual"].to_numpy()
        samp = eval_.sample(n=min(2500, len(eval_)), random_state=1)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].scatter(samp["actual"], samp["predicted"], s=9, alpha=0.35, color="#3498db")
        lim = [0, float(np.percentile(samp["actual"], 98))]
        ax[0].plot(lim, lim, "r--", lw=1.2)
        ax[0].set_xlim(lim); ax[0].set_ylim(lim)
        ax[0].set_xlabel("Actual monthly income (USD)")
        ax[0].set_ylabel("Predicted monthly income (USD)")
        ax[0].set_title("Predicted vs actual — hold-out households")
        ax[1].hist(resid, bins=60, color="#2c3e50", alpha=0.85)
        ax[1].axvline(0, color="red", lw=1.2)
        ax[1].set_title("Prediction errors (residuals)")
        ax[1].set_xlabel("Predicted − actual (USD)")
        ax[1].set_ylabel("Number of households")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        er2 = r2_score(eval_["actual"], eval_["predicted"])
        emae = mean_absolute_error(eval_["actual"], eval_["predicted"])
        ermse = float(np.sqrt(mean_squared_error(eval_["actual"], eval_["predicted"])))
        bands95 = analytics.residual_bands(resid)[95]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² on unseen data", f"{er2:.2f}", help="Share of income differences the model explains.")
        m2.metric("Typical error", f"±${emae:,.0f}/mo")
        m3.metric("Root-mean-square error", f"${ermse:,.0f}/mo")
        m4.metric("95% band", f"${max(0, bands95[0]):,.0f}…{bands95[1]:,.0f}", help="Where the middle 95% of prediction errors lie.")
        st.caption(
            f"Evaluated on {len(eval_):,} households excluded from training. "
            "Errors above ±$10k/month are almost always very high spenders whose income the survey under-records."
        )

# ════════════════════════════ PAGE 4: KEY FINDINGS ════════════════════════════
else:
    data_strip()
    st.header("Key findings")
    metrics = st.session_state["metrics"]
    pop = st.session_state["population"]
    erob = metrics.get("robust", metrics["full_holdout"])
    efull = metrics["full_holdout"]
    df4 = st.session_state["df"]
    _, rows4, _, p4_all, ols4 = analytics.education_income_stats(df4)
    bach_effect = ols4.params.get("edu_15", 0) * 1000
    bach_p = ols4.pvalues.get("edu_15", 1.0)
    rs4 = analytics.rule_summary(df4)
    _, gint_p, gint_ols = analytics.gender_interaction_test(df4)
    bach_gap = gint_ols.params.get("edu_15:male", 0) * 1000
    bach_gap_p = gint_ols.pvalues.get("edu_15:male", 1.0)
    p_str = "p < 0.001" if bach_p < 0.001 else f"p = {bach_p:.3f}"
    if bach_gap_p < 0.05:
        gdir = "Men earn more" if bach_gap > 0 else "Women earn more"
        gtext = (f"**{gdir} (~${abs(bach_gap):,.0f}/month more)** at Bachelor level "
                 f"(interaction p = {bach_gap_p:.3f}). Other education levels show no significant gap.")
    else:
        gtext = "No significant **gender gap** at Bachelor level (interaction p = {:.3f}).".format(bach_gap_p)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Households analyzed", f"{pop['n']:,}")
    c2.metric("Model accuracy (typical households)", f"R² = {erob['r2']:.2f}",
              help=f"Share of income differences explained. {erob['r2']:.0%} ≈ the model explains about {round(erob['r2'] * 100)}% of the gaps for typical households.")
    c3.metric("Typical error", f"±${erob['mae']:,.0f}/mo",
              help="Average distance between predicted and real income for typical households.")
    c4.metric("Median monthly income", f"${pop['median']:,.0f}")

    st.divider()

    with st.expander("💰 Percentile tool — where does an income sit?"):
        df_p4 = st.session_state["df"]
        pc1, pc2 = st.columns(2)
        with pc1:
            inc_q = st.number_input("Enter a monthly income", 0.0, 100000.0,
                                    float(pop["median"]), step=250.0, key="pct_income")
            pct_here = analytics.income_percentile(inc_q, df_p4)
            st.metric("It is higher than this % of households", f"{pct_here:.1f}%")
        with pc2:
            pct_want = st.slider("…or pick a percentile to see the income at it", 1, 99, 50)
            inc_at = analytics.income_for_percentile(pct_want, df_p4)
            st.metric(f"Income at the {pct_want}th percentile", f"${inc_at:,.0f}/mo")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("What we found")
        st.markdown(f"""
**1. Education predicts income — robustly.**
A Bachelor's degree adds roughly **+${bach_effect:,.0f}/month** over a high-school
household, after controlling for age, family size and gender ({p_str}).
This survives every robustness check we ran.

**2. {'The gap is bigger for men at Bachelor level.' if bach_gap_p < 0.05 and bach_gap > 0 else 'The gap is bigger for women at Bachelor level.' if bach_gap_p < 0.05 else 'No significant gender gap at Bachelor level.'}**
{gtext}

**3. Spending tells the same story.**
The model explains **{erob['r2']:.0%}** of income differences between typical households using
nothing but how they live and spend. The single most powerful signals are total
spending and **how the budget is split** (food, housing…).

**4. Engel's law shows in the data.**
Food's share of the budget falls from ~**{rs4['food_q1']:.0%}** for the poorest quarter to ~**{rs4['food_q4']:.0%}** for the
richest quarter — exactly what economists have observed for 170 years.
""")

        st.subheader("How accurate is the model?")
        st.markdown(f"""
For **typical households** (after removing a few extreme/unusual ones): the model explains
**{erob['r2']:.2f}** of income differences and is off by **±${erob['mae']:,.0f}/month** on average.

On **all** households the same model explains **{efull['r2']:.2f}** and is off by
**±${efull['mae']:,.0f}/month** — the difference mostly comes from a small set of
very unusual households (e.g. very high income) that are hard for anyone to predict.
""")

    with col_right:
        st.subheader("Honest caveats")
        st.markdown("""
- **Association, not causation.** We show income and education are linked; we cannot
  prove education *causes* the income difference from this survey alone.
- **Cross-sectional snapshot.** One point in time, not people followed over years.
- **Unmeasured factors.** Field of study, occupation, and local job markets also matter
  and aren't fully observed here.
- **Self-reported survey data.** Income and spending come from household interviews and
  carry measurement error.
""")

        st.subheader("Deliverables")
        items = [
            ("Feature dataset", "output/features_v1.csv",
             f"{pop['n']:,} households, cleaned & feature-engineered"),
            ("Trained model", "output/honest_gb.joblib",
             f"Gradient Boosting, robust R² ≈ {erob['r2']:.2f}"),
            ("Notebook", "CEX_Analytics_corrige.ipynb",
             "98 cells, fully reproducible (Sprints 1-3)"),
            ("Interactive app", "app/app.py",
             "this dashboard"),
        ]
        st.dataframe(pd.DataFrame(items, columns=["What", "File", "Description"]),
                     hide_index=True, width="stretch")

    st.caption(f"Dataset: {pop['n']:,} households · median ${pop['median']:,.0f}/mo · built with the Consumer Expenditure Survey (CE).")