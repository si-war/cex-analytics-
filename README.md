# 📊 CEXInsight — Education, Income & Household Spending (CE-PUMD)

A complete machine-learning project on the **U.S. Consumer Expenditure Survey (CE)**, made of two
parts that share the same data:

1. **The Jupyter notebook** (`CEX_Analytics_corrige.ipynb`) — the full research: data cleaning,
   feature engineering, model comparison, statistical tests, and the education→income study.
2. **The Streamlit app** (`app/`) — the same analysis, presented interactively, with a
   ready-to-use pre-trained model.

This guide is written for a laptop with **only VS Code installed** — every step is explained.

---

## 1. What is inside the folder

| Item | What it is |
|------|-----------|
| `CEX_Analytics_corrige.ipynb` | The notebook to run (Sprints 1 → 3) |
| `app/` | The Streamlit dashboard (Python source files) |
| `output/` | Pre-computed results: **`features_v1.csv`** + **`honest_gb.joblib`** (trained model) |
| `dataset used/` | The raw survey data, two subfolders: `fmli/` and `memi/` |
| `requirements.txt` | List of Python packages to install |
| `README.md` | This guide |

> ⚠️ **Keep the folder structure intact.** The app reads `output/features_v1.csv` and
> `output/honest_gb.joblib` automatically. Do not move `output/` or `app/` out of the project folder.

---

## 2. On your laptop — install the tools (once)

### 2.1 Install Python (3.10 or 3.11)

1. Go to <https://www.python.org/downloads/> and download the latest **3.11** installer
   (e.g. `python-3.11.x-amd64.exe`).
2. Run it. **Important: tick the box “Add python.exe to PATH”** at the bottom of the first
   screen, then click *Install Now*.

To verify, open a **new** terminal and type:

```bash
python --version
```

You should see something like `Python 3.11.9`. If the message is `'python' is not recognized…`,
Python was not added to the PATH (see [Troubleshooting](#7-troubleshooting)).

### 2.2 Install VS Code extensions

Open VS Code, open the **Extensions** panel (icon with 4 squares on the left, or `Ctrl+Shift+X`)
and install these two:

- **Python** (from Microsoft) — `ms-python.python`
- **Jupyter** (from Microsoft) — `ms-toolsai.jupyter`

### 2.3 Open the project in VS Code

1. In VS Code: **File → Open Folder…** and select the project folder.
2. Open the terminal: **Terminal → New Terminal** (`Ctrl+Shift+ò` / `` Ctrl+` ``).
   From now on, **run every command in this terminal**.

---

## 3. One-time setup — install the Python packages

In the VS Code terminal (the terminal starts in the project folder automatically):

```bash
pip install -r requirements.txt
```

This installs everything: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy,
statsmodels, jupyter, and streamlit. It downloads a few hundred MB the first time — be patient.

> If you get a “permission denied” error, instead run:
> `python -m pip install --user -r requirements.txt`

When it finishes, move on to Part A.

---

## 4. Part A — Run the notebook

1. In VS Code, open `CEX_Analytics_corrige.ipynb` (click the file in the Explorer).
2. Top-right of the notebook, click the **kernel selector** and choose:
   `Python 3.11 …` (the interpreter you installed).
   If asked, click **“Select Kernel” → “Python Environments”** and pick it.
3. Click **“Run All”** (the `▶▶` button at the top of the notebook) — or run cells one by one
   with `Shift+Enter`.

### The two prompts you MUST answer

The notebook stops twice and asks you to paste a folder path (`input()` box below the cell).
**Both times, paste the full path of the `dataset used` folder** — the one that contains the
`fmli` and `memi` subfolders.

First prompt (Sprint 1, finds the `fmli*.csv` files):

```
Paste your dataset folder path:  C:\Users\<YourName>\Desktop\stage esprit-4ds\dataset used
```

Second prompt (Sprint 3, finds the `memi*.csv` files — paste the **same** path again):

```
Paste your dataset folder path:  C:\Users\<YourName>\Desktop\stage esprit-4ds\dataset used
```

> On Windows you can copy the path from the Explorer address bar. No quotes needed.

The notebook then runs the whole pipeline and saves its results into the `output/` folder.

---

## 5. Part B — Run the Streamlit app

From the same VS Code terminal, run:

```bash
python -m streamlit run app/app.py
```

After a few seconds a browser window opens at **http://localhost:8501** with the app.
Keep the terminal window open while you use it (closing it stops the app).

> If the browser does not open automatically, open that address manually in Chrome/Edge.

### How the app works (4 pages, left-hand menu)

| Page | What you can do |
|------|-----------------|
| **1 · Introduction & data** | Read the project intro. Choose the **“Use the built-in sample”** (17,339 US households) **or** switch the radio to **“Upload your own files”** and drop your `fmli*.csv` files — the app re-trains the model on **your** data and every other page updates. |
| **2 · Predict income & the rules** | Describe a household (age, gender, region, family, spending…) → predicted monthly income, “richer than X%”, your budget vs the **50/30/20** rule, and a live check of **Engel's law / 30% housing** rules. |
| **3 · Education & income** | Seven tabs: predict income from an education level, income by education, the gender gap, feature importance, household clustering, spending profiles, and model accuracy. |
| **4 · Key findings** | The honest summary of the results and caveats, plus a percentile tool. |

**Loading your own data (optional).** On page 1, select “Upload your own files” in the radio,
then drop all your `fmli*.csv` files into the upload box and wait ~1 minute while the app trains
a new model on them.

---

## 6. How the two parts connect

```
CEX notebook  ──writes──►  output/features_v1.csv   ⇦ the Streamlit "built-in sample" data
                          output/honest_gb.joblib   ⇦ the pre-trained model the app uses
```

- The app **works immediately after setup** — no need to run the notebook first.
- If the model file is missing, the app is smart: it trains the model itself the first time
  (a spinner appears for ~2 minutes) using `features_v1.csv`.

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `'python' is not recognized as an internal or external command` | Python was not added to PATH. Re-run the Python installer, choose *Modify*, and tick **“Add python.exe to PATH”**. Or use `py` instead of `python` (`py -m pip install …`). |
| `'pip' is not recognized` | Use the full command: `python -m pip install …`. |
| `pip install` permission denied | `python -m pip install --user -r requirements.txt` |
| VS Code does not list a Python kernel | Open the notebook, click the kernel name, **“Select Kernel” → “Python Environments”** and choose the interpreter. If none appears, restart VS Code. |
| `streamlit: command not found` | Use `python -m streamlit run app/app.py` |
| Port 8501 already in use | `python -m streamlit run app/app.py --server.port 8502` |
| The app trains the model for the first time (~2 min) | Normal when `output/honest_gb.joblib` is absent — it means the model file did not arrive with the folder. It only happens once. |
| “No files yet” / upload does nothing | On page 1 use the radio **“Upload your own files”** first, then the drop zone appears. |
| First notebook run installs packages automatically | The notebook self-installs its own dependencies if missing — let it finish. |
| `ModuleNotFoundError: statsmodels` / `seaborn` | Re-run `pip install -r requirements.txt` and restart the kernel. |

---

## 8. Expected results (to check it worked)

- **Notebook:** output files appear in `output/` (e.g. `features_v1.csv`, `honest_gb.joblib`),
  and the final cell prints the conclusion of the education→income study.
- **App:** page 2 predicts an income; page 3 shows income differences across education levels
  (Bachelor ≈ **+$3,900 to +$4,600/month** over High School, statistically significant);
  page 4 reports model accuracy **R² ≈ 0.72–0.77** for typical households.
- **Engel's law:** food's share of the budget falls from ~**22%** (poorest quarter) to
  ~**15%** (richest quarter).