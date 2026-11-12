# Baseball Hitting 2015–2024

An end-to-end analysis of MLB position-player batting stats from 2015 to 2024 (Baseball Reference data). The project covers cleaning the data, exploratory analysis, hypothesis testing, and regression models that predict a player's WAR (Wins Above Replacement).

## Repository Contents

| File | Description |
|---|---|
| `Baseball Hitting.ipynb` | Main notebook: cleaning, EDA, hypothesis testing and WAR regression models |
| `Baseball Hitting.py` | Python script export of the notebook |
| `cleaned_batting_stats.csv` | Batting stats dataset (4,502 player-seasons) |

## Commit History

### `6e628bd`: Initial commit (2026-06-02)
- Created the repository with a `.gitattributes` file.

### `aa6fa57`: Commit #1, data cleaning and start of EDA (2026-06-02)
- **Added the dataset:** `cleaned_batting_stats.csv`.
- **Cleaned the data in the notebook:**
  - dropped the `Rk`, `RowNum` and `Pos` columns
  - converted counting stats (G, PA, AB, H, HR, RBI, etc.) to integers
- **Fixed player names:** repaired garbled characters and accents, and removed `*`, `#` and extra whitespace.
- **Created true/false award columns** (Rookie of the Year, All-Star, MVP, Silver Slugger, Gold Glove, award consideration) and a "Played For Multiple Teams" flag.

### `be7d288`: Commit #2, data pipeline fixes, line graphs and pivot tables (2026-06-09)
- **Fixed the data pipeline:**
  - players traded mid-season (`2TM`/`3TM`/`4TM`) now keep only their combined season row, removing duplicate team rows
  - award parsing now uses exact matches (e.g. `ROY-1` no longer matches `ROY-10`)
- **Added EDA:**
  - position players per team and an age histogram
  - pivot tables of career home run and strikeout leaders, All-Star appearances and MVP winners
  - line graphs of league-wide totals by year
- **Added `Baseball Hitting.py`**, a script version of the notebook.

### `c0c7cd1`: Commit #3, finished EDA, more pipeline fixes, hypothesis testing (2026-06-26)
- **Rewrote the name cleaning and award parsing** into a full list of each player's awards.
- **Award analysis:**
  - Mookie Betts's award breakdown and the players with the most awards
  - Silver Slugger, Gold Glove and All-Star counts by year
  - which teams are strong or weak
- **Hypothesis test (two-sample t-test):** players who stayed with one team averaged a higher WAR (1.43) than players who played for multiple teams (0.55), p = 4.7e-22, Cohen's d = 0.48.

### `0bbb2fc`: Commit #4, baseline regression models (2026-07-16)
- **Baseline model:** a linear regression pipeline to predict WAR from batting stats and award flags, cross-validated R² = **0.806**.
- **Coefficients:** plotted the model's coefficients.
- **Trimmed feature set:** tested a smaller set of features, which slightly lowered R² (to 0.793).
- **Random forest:** started a random forest model.

### `dbb05bc`: Commit #5, finished the regression pipeline, XGBoost performs best (2026-07-21)
- **Compared six models with k-fold cross-validation:**

  | Model | CV RMSE | CV MAE | CV R² |
  |---|---|---|---|
  | Linear Regression | 0.823 | 0.624 | 0.806 |
  | Ridge | 0.823 | 0.625 | 0.806 |
  | Lasso | 0.838 | 0.642 | 0.799 |
  | Random Forest | 0.818 | 0.620 | 0.808 |
  | **XGBoost** | **0.801** | **0.608** | **0.816** |
  | LightGBM | 0.802 | 0.605 | 0.816 |

- **Tuned XGBoost** with `RandomizedSearchCV`, raising the cross-validated R² to **0.821**.
- **Checked for overfitting.**
- **Tested a trimmed 20-feature XGBoost model:** R² = 0.814, slightly worse, so the full feature set was kept.

### `8240a06`: Commit #6 (2026-10-26)
- Re-exported `Baseball Hitting.py` from the latest notebook, so the script matches the final cleaning, EDA and modeling code.

## Tools
Python, pandas, NumPy, SciPy, scikit-learn, XGBoost, LightGBM, matplotlib, seaborn, Jupyter Notebook
