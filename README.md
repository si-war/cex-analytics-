# CEX Analytics — Education, Income & Household Spending

Machine learning project analyzing the **Consumer Expenditure Survey (CE-PUMD)** data.
Explores how education, demographics, and spending patterns relate to household income,
using statistical tests, clustering, and ensemble models.

## Repository Structure

```text
## Repository Structure

```text
.
├── CEX_Analytics_final.ipynb        ← Final CEX analysis notebook
├── app/
│   ├── app.py                        ← Streamlit web application
│   └── preprocessing.py              ← Data preprocessing & prediction logic
├── dataset used/
│   ├── fmli/
│   │   ├── fmli232.csv
│   │   ├── fmli233.csv
│   │   ├── fmli234.csv
│   │   ├── fmli241.csv
│   │   ├── fmli241x.csv
│   │   ├── fmli242.csv
│   │   ├── fmli243.csv
│   │   ├── fmli244.csv
│   │   └── fmli251.csv
│   └── memi/
│       ├── memi222.csv
│       ├── memi223.csv
│       ├── memi224.csv
│       └── memi231.csv
└── README.md

## Prerequisites

- Python 3.11+
- [VS Code](https://code.visualstudio.com/) with the **Python** and **Jupyter** extensions
  (install: `code --install-extension ms-python.python`, `code --install-extension ms-toolsai.jupyter`)
- CE-PUMD data files (see [Dataset](#dataset) below)

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register the Jupyter kernel for VS Code
python -m ipykernel install --user --name python311
```

## Dataset

This project uses the **CE-PUMD Interview Survey** microdata from the U.S. Bureau of Labor Statistics.

You need **10 CSV files** (5 fmli + 5 memi) covering quarters 2024Q1 through 2025Q1:

| File | What it contains |
|------|-----------------|
| `fmli241x.csv` | Family characteristics & income — 2024 Q1 |
| `fmli242.csv` | Family characteristics & income — 2024 Q2 |
| `fmli243.csv` | Family characteristics & income — 2024 Q3 |
| `fmli244.csv` | Family characteristics & income — 2024 Q4 |
| `fmli251.csv` | Family characteristics & income — 2025 Q1 |
| `memi241x.csv` | Member education — 2024 Q1 |
| `memi242.csv` | Member education — 2024 Q2 |
| `memi243.csv` | Member education — 2024 Q3 |
| `memi244.csv` | Member education — 2024 Q4 |
| `memi251.csv` | Member education — 2025 Q1 |

Download from the [BLS CE-PUMD website](https://www.bls.gov/cex/pumd_data.htm).
Place all files in a single folder on your machine.

## Running the Notebook

```bash
code CEX_Analytics_corrige.ipynb
```

1. Select the `python311` kernel (top-right of the notebook).
2. Run cells **top to bottom**.
3. **When prompted**, paste the **full path** to the folder containing your `fmli*.csv` and `memi*.csv` files.

Example prompt and expected input:

```
Paste your dataset folder path: C:\Users\you\Desktop\CEX_data\data needed
```

The notebook will:
- Auto-install any missing Python packages on the first run
- Load and merge 5 quarters of fmli and memi data
- Engineer features, train models, run statistical tests
- Save outputs to the `output/` folder (features_v1.csv, final_model.joblib, etc.)

> **Important:** You will be asked for the dataset path **twice** — once for the fmli files (Sprint 1)
> and once for the memi files (Sprint 3). Paste the same folder path both times.

## Running the Streamlit App

After running the notebook (which generates the `output/` files):

```bash
streamlit run app/app.py
```

Or from VS Code, run in the terminal:

```bash
python -m streamlit run app/app.py
```

The app opens at **http://localhost:8501** with three pages:

### 1. Prediction
Enter household demographics and spending → get a predicted monthly income with a
breakdown of which features push the prediction above or below the population median.

### 2. Analysis
Four interactive tabs:
- **Education-Income Gap** — Boxplot + gender interaction test (ANOVA + OLS)
- **Feature Importance** — Permutation importance from the Gradient Boosting model
- **Clustering** — 2D KMeans on education × income, revealing natural population tiers
- **Spending Profiles** — Budget shares by education level (Engel's law visualization)

### 3. Conclusions
Model performance summary (R², MAE, RMSE) + exportable HTML report.

## What the Notebook Covers

### Sprint 1 — Data Preparation (US-03 → US-07)
- Load 5 quarters of fmli + memi data via interactive path input
- Reconcile food variables (`FDHOMEPQ` → `GROCERPQ` transition in 2024Q2)
- Clean, engineer features (budget shares, OECD scale, per-capita spending)
- Encode categoricals, apply RobustScaler, run PCA

### Sprint 2 — Model Building (US-08 → US-12)
- Compare 6 models (Linear, Ridge, Lasso, RF, GB, SVR)
- Hyperparameter tuning via GridSearchCV
- Stacking ensemble (R² = 0.987)
- Residual diagnostics

### Sprint 3 — Education-Income Analysis (US-13 → US-17)
- ANOVA + Tukey HSD for education groups
- OLS regression with HC3 robust standard errors
- Permutation importance (corrected, no target leakage)
- 2D and full-feature clustering
- MEMI member-level analysis (education info from member data)
- Gender × education interaction, age cohorts, spending profiles
- Synthesis and conclusions

## Output Files

| File | Description |
|------|-------------|
| `output/features_v1.csv` | 62-column feature table (demographics + spending + encoded) |
| `output/processed_data_v1.csv` | Scaled features + `FINCBTXM_M` target (for ML) |
| `output/clean_data_v1.csv` | Cleaned raw data before feature engineering |
| `output/final_model.joblib` | Serialized stacking ensemble + metadata |

These are generated by the notebook. The Streamlit app reads them directly.

## Notes

- The notebook uses `input()` to get the dataset path — you must paste it when prompted.
- `mtbi*.csv` (transaction-level expenditure data) is **not used** — it contains no education information.
- The `housing_income_ratio` feature was removed to prevent target leakage.
- `FDHOMEPQ` only exists in 2024Q1; from 2024Q2 onward, `GROCERPQ` is used instead.
- `TFOODTOP`, `TFOODHOP`, `TFOODAWP` are near-zero noise and excluded.
- All income values are converted to **monthly USD** (`annual / 12`).
