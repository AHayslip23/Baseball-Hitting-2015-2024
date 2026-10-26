#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# In[2]:


baseball = pd.read_csv("cleaned_batting_stats.csv")
cleaned_baseball = baseball.copy()


# In[3]:


cleaned_baseball.head(5)


# In[4]:


cleaned_baseball.columns


# In[5]:


cleaned_baseball.dtypes


# **data cleaning**
# <br>drop columns of Rk and RowNum</br>

# In[6]:


cleaned_baseball = cleaned_baseball.drop(columns=["Rk", "RowNum", "Pos"])
cleaned_baseball


# **Check Data Types for Numeric Columns**

# Age       float64
# WAR       float64
# G         float64
# PA        float64
# AB        float64
# R         float64
# H         float64
# 2B        float64
# 3B        float64
# HR        float64
# RBI       float64
# SB        float64
# CS        float64
# BB        float64
# SO        float64
# OPS+      float64
# TB        float64
# GIDP      float64
# HBP       float64
# SH        float64
# SF        float64
# IBB       float64

# In[7]:


numerics = ["Age", "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "OPS+", "TB", "GIDP", "HBP", "SH", "SF", "IBB", "Rbat+"]


# In[8]:


cleaned_baseball[numerics] = cleaned_baseball[numerics].astype("int64")
cleaned_baseball.head(5)


# **Create booleans for awards**

# **Remove Weird Asterisks and Funky Naming Conventions**
# <br>-remove # </br>
# <br>-remove accents </br> 
# <br>-remove weird encoding </br>
# <br>-remove *
# <br>-remove whitespace </br>

# In[9]:


import unicodedata
import re

def fix_mojibake(s):
    """Fix UTF-8 bytes misinterpreted as Latin-1"""
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s

def strip_accents(s):
    """Remove diacritical marks (á → a, ñ → n, etc)"""
    if not isinstance(s, str):
        return s
    nfd = unicodedata.normalize('NFD', s)
    return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')

def clean_player_name(s):
    """Comprehensive player name cleaning"""
    if not isinstance(s, str):
        return s

    # Step 1: Fix mojibake FIRST (before accent stripping)
    s = fix_mojibake(s)

    # Step 2: Strip accents
    s = strip_accents(s)

    # Step 3: Remove TRAILING # and * only (use $ anchor)
    s = re.sub(r'[#*]+$', '', s)

    # Step 4: Clean other unexpected special characters
    s = re.sub(r'[^\w\s\-.]', '', s)

    # Step 5: Normalize whitespace
    s = ' '.join(s.split())

    return s

cleaned_baseball['Player'] = cleaned_baseball['Player'].apply(clean_player_name)


# In[10]:


# Step 4: Remove duplicate team rows FIRST ← key fix
def remove_duplicate_team_rows(df):
    multi_team = df[df["Team"].str.contains("TM", na=False)][["Player", "Year"]]
    multi_team["is_multi"] = True
    df = df.merge(multi_team, on=["Player", "Year"], how="left")
    df = df[~((df["is_multi"] == True) & (~df["Team"].str.contains("TM", na=False)))]
    df = df.drop(columns=["is_multi"])
    return df

cleaned_baseball = remove_duplicate_team_rows(cleaned_baseball)


# In[11]:


cleaned_baseball["Rookie Of The Year"] = np.where(cleaned_baseball["Awards"].str.contains(r"ROY-1(?!\d)", na=False, regex=True), 1, 0).astype(bool)
cleaned_baseball["All Star Appearance"] = np.where(cleaned_baseball["Awards"].str.contains("AS", na=False), 1, 0).astype(bool)
cleaned_baseball["Gold Glove"] = np.where(cleaned_baseball["Awards"].str.contains("GG", na=False, regex=True), 1, 0).astype(bool)
cleaned_baseball["MVP"] = np.where(cleaned_baseball["Awards"].str.contains(r"MVP-1(?!\d)", na=False, regex=True), 1, 0).astype(bool)
cleaned_baseball["Sliver Slugger"] = np.where(cleaned_baseball["Awards"].str.contains("SS", na=False), 1, 0).astype(bool)
cleaned_baseball["Considered for Rookie Of The Year"] = np.where(cleaned_baseball["Awards"].str.contains("ROY", na=False), 1, 0).astype(bool)
cleaned_baseball["Considered for MVP"] = np.where(cleaned_baseball["Awards"].str.contains(r"MVP(?!\d)", na=False, regex=True), 1, 0).astype(bool)


# **Create Boolean for Players who played for more than one team**

# In[12]:


multiple = ["2TM", "3TM", "4TM"]
cleaned_baseball["Played For Multiple Teams"] = np.where(cleaned_baseball["Team"].isin(multiple), 1, 0).astype(bool)


# In[13]:


cleaned_baseball.dtypes


# **Which team employed the most amount of position players**
# - records with 2TM, 3TM and 4TM are removed for this analysis
# - smaller markets like tampa bay, minnesota and cleveland are towards the top as they don't have the funds to keep players when their contracts are up

# In[14]:


players_without_multiple = cleaned_baseball[~cleaned_baseball["Team"].isin(multiple)]


# In[15]:


order = players_without_multiple["Team"].value_counts().index
plt.figure(figsize=(16, 5))
sns.countplot(data=players_without_multiple, x="Team", order=order, hue="Team")
plt.title("Count of Unique Number of Position Players Employed")
plt.xlabel("Team Name")
plt.ylabel("Unique Count of Players Employed")
plt.show()


# In[16]:


order1 = cleaned_baseball["Age"].value_counts().index
plt.figure(figsize=(7, 3))
plt.title("Histogram of Players By Their Age")
plt.hist(data=cleaned_baseball, x="Age", bins=15, edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Unique Count of Players")
plt.show()


# In[17]:


homers = pd.pivot_table(cleaned_baseball, index="Player", values="HR", aggfunc="sum")
top_1o_pivot = homers.nlargest(10, "HR")
print(top_1o_pivot)


# In[18]:


strikeouts = pd.pivot_table(cleaned_baseball, index="Player", values="SO", aggfunc="sum")
most_strikeout_prone = strikeouts.nlargest(10, "SO")
print(most_strikeout_prone)


# In[19]:


allstar = cleaned_baseball[cleaned_baseball["All Star Appearance"] == True]

allstarappearance = pd.pivot_table(allstar, index="Player", values="All Star Appearance", aggfunc="sum")
most_all_stars = allstarappearance.nlargest(10, "All Star Appearance")
print(most_all_stars)


# In[20]:


MVP = cleaned_baseball[cleaned_baseball["MVP"] == True]

MVPWins = pd.pivot_table(MVP, index="Player", values="MVP", aggfunc="sum")
most_mvps = MVPWins.nlargest(10, "MVP")
print(most_mvps)


# In[21]:


home_runs_totals = cleaned_baseball.groupby("Year")["HR"].sum()

home_runs_totals.plot(kind="line", marker="o", color="blue", linewidth=2)
plt.title("Trend of Home Runs By Season")
plt.xlabel("Year")
plt.ylabel("Total Number of Home Runs")
plt.grid(True)
plt.show()


# In[22]:


sandiegopadres = cleaned_baseball[cleaned_baseball["Team"] == "SDP"]
padres_homers = sandiegopadres.groupby("Year")["HR"].sum()

padres_homers.plot(kind="line", marker="o", color="brown", linewidth=2)
plt.title("Trend of Home Runs By Season (San Diego Padres)")
plt.xlabel("Year")
plt.ylabel("Total Home Runs")
plt.grid(True)
plt.show()


# In[23]:


mannyandbryce = cleaned_baseball[cleaned_baseball["Player"].isin(["Manny Machado", "Bryce Harper"])]

home_runs = mannyandbryce.groupby(["Player", "Year"])["HR"].sum().unstack("Player")

home_runs.plot(kind="line", marker="o", linewidth=2)
plt.title("Trend of Home Runs By Season: Bryce Harper vs Manny Machado")
plt.xlabel("Year")
plt.ylabel("Total Home Runs")
plt.legend(title="Player")
plt.grid(True)
plt.show()


# In[24]:


def parse_awards_complete(award_string):
    if pd.isna(award_string):
        return []

    awards_list = []
    for award in award_string.split(','):
        award = award.strip()

        if award == 'MVP-1':
            awards_list.append('MVP')
        elif award.startswith('MVP'):
            continue
        elif award == 'ROY-1':
            awards_list.append('ROY')
        elif award.startswith('ROY'):
            continue
        else:
            awards_list.append(award)

    return sorted(awards_list)

cleaned_baseball['awards'] = cleaned_baseball['Awards'].apply(parse_awards_complete)


# In[25]:


award_winners = cleaned_baseball[
    cleaned_baseball['awards'].apply(
        lambda x: any(award in x for award in ['MVP', 'ROY', 'AS', 'GG', 'SS'])
    )
]


# **What is the composition of awards won by mookie betters**

# In[26]:


player = "Mookie Betts"

player_data = award_winners[award_winners["Player"] == player]

all_awards = []
for awards_list in player_data["awards"]:
    all_awards.extend(awards_list)

print(f"{player} - Award Composition:")
print(f"  MVP:        {all_awards.count('MVP')}")
print(f"  ROY:        {all_awards.count('ROY')}")
print(f"  All-Star:   {all_awards.count('AS')}")
print(f"  Gold Glove: {all_awards.count('GG')}")
print(f"  Slugger:    {all_awards.count('SS')}")
print(f"  {'─'*30}")
print(f"  Total:      {len(all_awards)}")


# **Which Players Have Accumulated the most amount of awards**

# In[27]:


top_10_players = award_winners.groupby("Player").size().sort_values(ascending=False).head(10)

total_awards = award_winners.groupby("Player")["awards"].apply(
    lambda x: sum(len(awards) for awards in x)
).sort_values(ascending=False).head(10)

print("\nTop 10 Players - Total Individual Awards:")
print(total_awards)


# In[28]:


sliver_slugger = cleaned_baseball[cleaned_baseball["awards"].apply(lambda x: "SS" in x)]

ss_by_year = sliver_slugger.groupby("Year").size()

plt.figure(figsize=(9, 5))
plt.plot(ss_by_year.index, ss_by_year.values, marker='o', linewidth=2.5, markersize=8, color='steelblue')
plt.xlabel("Season", fontsize=12)
plt.ylabel("Total Number of Recipents", fontsize=12)
plt.title("Number of Silver Slugger Award Winners Per Season", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[29]:


gold_glove = cleaned_baseball[cleaned_baseball["awards"].apply(lambda x: "GG" in x)]

gold_glove_by_year = gold_glove.groupby("Year").size()

plt.figure(figsize=(11, 5))
plt.plot(gold_glove_by_year.index, gold_glove_by_year.values, marker="o", linewidth=2.0, markersize=8, color="steelblue")
plt.title("Total Amount of Gold Glove Winners")
plt.ylabel("Total Amount of Recipents")
plt.xlabel("Season")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()


# In[30]:


all_star = cleaned_baseball[cleaned_baseball["awards"].apply(lambda x: "AS" in x)]

all_star_by_year = all_star.groupby("Year").size()

plt.figure(figsize=(9, 5))
plt.plot(all_star_by_year.index, all_star_by_year.values, marker="o", linewidth=2, markersize=8, color="steelblue")
plt.xlabel("Season")
plt.ylabel("Total Number of Particpants")
plt.title("Total Number of All Star Games Particpants Per Season")
plt.grid(True, alpha=0.1)
plt.tight_layout()
plt.show()


# In[31]:


cleaned_baseball[["G", "WAR"]].corr()


# In[32]:


sns.scatterplot(data=cleaned_baseball, x="G", y="WAR")
plt.title("Is there a correlation between Games Played and WAR")
plt.show()


# In[33]:


cleaned_baseball[["AB", "OBP"]].corr()


# In[34]:


plt.title("Is there a correlation between more At Bats and a Higher OBP?")
sns.scatterplot(data=cleaned_baseball, x="AB", y="OBP")
plt.show()


# In[35]:


cleaned_baseball[["BB", "OBP"]].corr()


# In[36]:


sns.scatterplot(data=cleaned_baseball, x="BB", y="OBP")
plt.show()


# In[37]:


plt.title("What WAR generally results in all star appearances?")
sns.boxplot(data=cleaned_baseball, x="All Star Appearance", y="WAR", hue="All Star Appearance")
plt.legend().remove()
plt.show()


# In[38]:


sns.boxplot(data=cleaned_baseball, x="Rookie Of The Year", y="WAR", hue="Rookie Of The Year")
plt.title("What WAR would qualify you as a rookie of the year winner")
plt.legend().remove()
plt.show()


# In[39]:


sns.boxplot(data=cleaned_baseball, x="MVP", y="WAR", hue="MVP")
plt.title("Which WAR would get you an MVP award?")
plt.legend().remove()
plt.show()


# In[40]:


cleaned_baseball.head(5)


# In[41]:


single_team = cleaned_baseball[cleaned_baseball["Played For Multiple Teams"] == False]
multiple_team = cleaned_baseball[cleaned_baseball["Played For Multiple Teams"] == True]

for stat in ["WAR", "R", "H", "2B", "3B", "HR", "RBI", "SB", "BB", "SO", "BA", "OBP", "SLG", "OPS", "OPS+", "TB", "GIDP"]:
    if stat in cleaned_baseball.columns:
        single_means = single_team[stat].mean()
        multi_means = multiple_team[stat].mean()

        print(f"\n{stat}:")
        print(f"  Single Team:     {single_means:.4f}")
        print(f"  Multiple Teams:  {multi_means:.4f}")
        print(f"  Difference:      {single_means - multi_means:.4f}")


# In[42]:


print(multiple_team.shape)


# In[43]:


single_war = single_team["WAR"].dropna()
multi_war = multiple_team["WAR"].dropna()

print(f"\nSingle Team WAR:     Mean={single_war.mean():.3f}, Std={single_war.std():.3f}, N={len(single_war)}")
print(f"Multiple Teams WAR:  Mean={multi_war.mean():.3f}, Std={multi_war.std():.3f}, N={len(multi_war)}")

from scipy import stats
t_stat, p_value = stats.ttest_ind(single_war, multi_war)

print(f"\nNull Hypothesis (H0): No difference in WAR between groups")
print(f"Alternative Hypothesis (H1): Single-team players have higher WAR")

print(f"\nTest Results:")
print(f"  t-statistic:        {t_stat:.4f}")
print(f"  p-value (two-tailed): {p_value:.2e}")
print(f"  p-value (one-tailed): {p_value/2:.2e}")

print("\n3. STATISTICAL SIGNIFICANCE")
print("-" * 80)

alpha = 0.05
if p_value < alpha:
    print(f"\n✓ RESULT: STATISTICALLY SIGNIFICANT (p = {p_value:.2e} < {alpha})")
    print(f"\nWe REJECT the null hypothesis.")
    print(f"There IS a significant difference in WAR between single-team")
    print(f"and multi-team players during the season.")
else:
    print(f"\n✗ RESULT: NOT STATISTICALLY SIGNIFICANT (p = {p_value:.4f} >= {alpha})")
    print(f"\nWe FAIL TO REJECT the null hypothesis.")
    print(f"There is NO significant difference in WAR between single-team")
    print(f"and multi-team players during the season.")

    print("\n4. EFFECT SIZE (COHEN'S D)")
    print("-" * 80)

n1 = len(single_war)
n2 = len(multi_war)

# Corrected pooled standard deviation
pooled_std = np.sqrt(
    ((n1 - 1) * single_war.std()**2 + 
     (n2 - 1) * multi_war.std()**2) / 
    (n1 + n2 - 2)
)

# Cohen's d
cohens_d = (single_war.mean() - multi_war.mean()) / pooled_std

print(f"\nPooled Standard Deviation: {pooled_std:.4f}")
print(f"Cohen's d:                {cohens_d:.4f}")

# Interpret effect size
if abs(cohens_d) < 0.2:
    effect_interpretation = "Negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "Small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "Medium"
else:
    effect_interpretation = "Large"

print(f"Effect Size Interpretation: {effect_interpretation}")


# **Which Teams Are Not Good and Which Teams are Good**

# In[44]:


single_team_only = cleaned_baseball[~cleaned_baseball['Team'].str.contains('TM', na=False)]


# In[45]:


team_impact = single_team_only.groupby("Team").agg({
    "WAR": ["mean", "median", "std"],
    "Player": "nunique"
}).round(3)

team_impact.columns = ["AVG_WAR", "Median_WAR", "std_WAR", "Unique_Players"]
team_impact = team_impact.sort_values("AVG_WAR", ascending=False)

print("\nTeam Impact on WAR (Top and Bottom Teams)")
print("="*80)
print("\nTOP 10 Teams (Career-Saving):")
print(team_impact.head(10))

print("\nBOTTOM 10 Teams (Career-Ruining):")
print(team_impact.tail(10))


# **Which Features Are Most Indicative of a Players War Value**
# - The base model returns an R2 Score of 80.61%

# In[46]:


cleaned_baseball.head(5)


# In[47]:


columns = cleaned_baseball[["Age", "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "OPS+", "TB", "GIDP", "HBP", "SH", "SF", "IBB", "Rbat+", "Rookie Of The Year", "All Star Appearance", "Gold Glove", "MVP", "Sliver Slugger", "Considered for Rookie Of The Year", "Considered for MVP", "Played For Multiple Teams", "WAR"]]


# In[48]:


columns.head(5)


# In[55]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd


# **Which models are the most effective and least effective at predicting R2 Score**
# - Linear Regression
# - Ridge 
# - Lasso 
# - Random Forest 
# - XG Boost
# - LightGBM

# In[61]:


features = [
    "Age", "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "OPS+", "TB", "GIDP", "HBP", "SH", "SF", "IBB", "Rbat+", "Rookie Of The Year", "All Star Appearance", "Gold Glove", "MVP", "Sliver Slugger", "Considered for Rookie Of The Year", "Considered for MVP", "Played For Multiple Teams" 
]

categorical_columns = ["Rookie Of The Year", "All Star Appearance", "Gold Glove", "MVP", "Sliver Slugger", "Considered for Rookie Of The Year", "Considered for MVP", "Played For Multiple Teams"]
numerical_columns = ["Age", "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "OPS+", "TB", "GIDP", "HBP", "SH", "SF", "IBB", "Rbat+"]


def make_preprocessor(scale_numeric=False):
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_columns),
            ("num", numeric_step, numerical_columns)
        ]
    )

def evaluate_model(df, name, model, scale_numeric=False, random_state=43):
    X = df[features]
    y = df["WAR"]

    pipeline = Pipeline(steps=[
        ("preprocessor", make_preprocessor(scale_numeric)),
        ("model", model)
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
    cv_rmse = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    cv_mae = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")

    return {
        "Dataset": name, 
        "CV_RMSE": -cv_rmse.mean(),
        "CV_MAE": -cv_mae.mean(),
        "CV_R2": cv_r2.mean()
    }

def run_all_models(df):
    model_configs = [
        ("Linear Regression", LinearRegression(), False),
        ("Ridge", Ridge(alpha=1.0, random_state=43), True),
        ("Lasso", Lasso(alpha=0.01, random_state=43), True),
        ("Random Forest", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=43, n_jobs=-1), False),
        ("XGBoost", XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=43, n_jobs=-1), False),
        ("LightGBM",LGBMRegressor(n_estimators=300, max_depth=-1, learning_rate=0.05, random_state=43, n_jobs=-1), False),

    ]

    results = [
        evaluate_model(df, name, model, scale_numeric=scale)
        for name, model, scale in model_configs
    ]
    return pd.DataFrame(results)

model_df = columns[features + ["WAR"]]
results_df = run_all_models(model_df)
print(results_df.round(4).to_string(index=False))


# **Optimize the XGBoost Model**
# 
# -Randomized Grid Searh CV bumps R2 score up to 82.1

# In[65]:


from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import uniform, randint
from xgboost import XGBRegressor 

X = model_df[features]
y = model_df["WAR"]

preprocessor = make_preprocessor(scale_numeric=False)

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(random_state=43, n_jobs=-1))
])

param_dist = {
    "model__n_estimators": randint(100, 800),
    "model__max_depth": randint(3, 8),
    "model__learning_rate": uniform(0.01, 0.19),   
    "model__subsample": uniform(0.6, 0.4),           
    "model__colsample_bytree": uniform(0.6, 0.4),     
    "model__min_child_weight": randint(1, 10),
    "model__reg_alpha": uniform(0, 1),                
    "model__reg_lambda": uniform(0.5, 2.5),          
    "model__gamma": uniform(0, 0.5),
}

kf = KFold(n_splits=5, shuffle=True, random_state=43)

search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=param_dist,
    n_iter=100, 
    scoring="r2",
    cv=kf,
    random_state=43,
    n_jobs=-1,
    verbose=1
)

search.fit(X, y)

print("Best CV R2:", search.best_score_)
print("Best params:", search.best_params_)


# **Check for overfitting**

# In[66]:


results = pd.DataFrame(search.cv_results_)
top5 = results.sort_values("mean_test_score", ascending=False).head(5)
print(top5[["mean_test_score", "std_test_score", "params"]])


# **Does the performance of the XGBBoost Model improve with feature optimiation**
# - no the model performs slighltly worse with feature optimization

# In[71]:


from sklearn.inspection import permutation_importance


# In[73]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=43)

xgb_pipeline.fit(X_train, y_train)

perm_result = permutation_importance(
    xgb_pipeline, X_test, y_test,
    scoring="r2",
    n_repeats=20,
    random_state=43,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": features,
    "r2_drop_mean": perm_result.importances_mean,
    "r2_drop_std": perm_result.importances_std
}).sort_values("r2_drop_mean", ascending=False)

print(perm_df.to_string(index=False))


# In[79]:


keep_features = [
    "Rbat+", "R", "Considered for MVP", "H", "2B", "OPS+", "SB", "PA",
    "Gold Glove", "HR", "G", "SH", "SO", "TB", "AB", "RBI",
    "All Star Appearance", "GIDP", "Age", "BB"
]

keep_categorical = ["Considered for MVP", "Gold Glove", "All Star Appearance"]
keep_numerical = [f for f in keep_features if f not in keep_categorical]

def make_preprocessor_trimmed(cat_cols, num_cols, scale_numeric=False):
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
            ("num", numeric_step, num_cols)
        ]
    )

def evaluate_xgboost_trimmed(df, feature_list, cat_cols, num_cols,
                              name="XGBoost (trimmed)", random_state=43,
                              model_params=None):
    X = df[feature_list]
    y = df["WAR"]

    if model_params is None:
        model_params = dict(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=random_state, n_jobs=-1
        )

    pipeline = Pipeline(steps=[
        ("preprocessor", make_preprocessor_trimmed(cat_cols, num_cols)),
        ("model", XGBRegressor(**model_params))
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
    cv_rmse = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    cv_mae = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error")
    return {
        "Dataset": name,
        "CV_RMSE": -cv_rmse.mean(),
        "CV_MAE": -cv_mae.mean(),
        "CV_R2": cv_r2.mean()
    }

keep_results = evaluate_xgboost_trimmed(
    model_df, keep_features, keep_categorical, keep_numerical,
    name="XGBoost (keep features)"
)
print(pd.DataFrame([keep_results]).round(4).to_string(index=False))


# **XGBoost**
# XGBoost Base: 0.8162 
# XGBoost Base + Randomized Grid Search: 0.821
# XGBoost Trimmed Features + K FOld Cross Validation 0.8136

# In[ ]:




