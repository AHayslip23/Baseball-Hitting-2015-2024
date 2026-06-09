#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# In[93]:


baseball = pd.read_csv("cleaned_batting_stats.csv")
cleaned_baseball = baseball.copy()


# In[94]:


cleaned_baseball.head(5)


# In[95]:


cleaned_baseball.columns


# In[96]:


cleaned_baseball.dtypes


# **data cleaning**
# <br>drop columns of Rk and RowNum</br>

# In[97]:


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

# In[98]:


numerics = ["Age", "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "OPS+", "TB", "GIDP", "HBP", "SH", "SF", "IBB", "Rbat+"]


# In[99]:


cleaned_baseball[numerics] = cleaned_baseball[numerics].astype("int64")
cleaned_baseball.head(5)


# **Create booleans for awards**

# In[100]:


cleaned_baseball["Rookie Of The Year"] = np.where(cleaned_baseball["Awards"].str.contains("ROY-1", na=False), 1, 0).astype(bool)
cleaned_baseball["All Star Appearance"] = np.where(cleaned_baseball["Awards"].str.contains("AS", na=False), 1, 0).astype(bool)
cleaned_baseball["MVP"] = np.where(cleaned_baseball["Awards"].str.contains("MVP-1", na=False), 1, 0).astype(bool)
cleaned_baseball["Sliver Slugger"] = np.where(cleaned_baseball["Awards"].str.contains("SS", na=False), 1, 0).astype(bool)
cleaned_baseball["Considered for Rookie Of The Year"] = np.where(cleaned_baseball["Awards"].str.contains("ROY", na=False), 1, 0).astype(bool)
cleaned_baseball["Considered for MVP"] = np.where(cleaned_baseball["Awards"].str.contains("MVP", na=False), 1, 0).astype(bool)


# **Create Boolean for Players who played for more than one team**

# In[101]:


multiple = ["2TM", "3TM", "4TM"]
cleaned_baseball["Played For Multiple Teams"] = np.where(cleaned_baseball["Team"].isin(multiple), 1, 0).astype(bool)


# **Remove Weird Asterisks and Funky Naming Conventions**
# <br>-remove # </br>
# <br>-remove accents </br> 
# <br>-remove weird encoding </br>
# <br>-remove *
# <br>-remove whitespace </br>

# In[102]:


import unicodedata

def fix_mojibake(s):
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s

def strip_accents(s):
    if not isinstance(s, str):
        return s
    return ''.join(
        c for c in unicodedata.normalize('NFKD', s)
        if not unicodedata.combining(c)
    )


# In[103]:


cleaned_baseball["Player"] = (
    cleaned_baseball["Player"]
        .apply(fix_mojibake)        # repair encoding first
        .apply(strip_accents)       # then strip diacritics
        .str.replace(r"[#*]", "", regex=True)  # then drop handedness markers
        .str.strip()                # tidy any leftover whitespace
)


# **Keep only the record with 2tmn/3tm etc....**

# In[104]:


# Fix multi-team seasons: keep only the combined "2TM"/"3TM" row, drop individual team rows
def remove_duplicate_team_rows(df):
    # Find players who have a multi-team row (2TM, 3TM) in a given year
    multi_team = df[df["Team"].str.contains("TM", na=False)][["Player", "Year"]]
    
    # Tag those Player+Year combos
    multi_team["is_multi"] = True
    df = df.merge(multi_team, on=["Player", "Year"], how="left")
    
    # Drop individual team rows for those players (keep only the TM row)
    df = df[~((df["is_multi"] == True) & (~df["Team"].str.contains("TM", na=False)))]
    
    # Drop the helper column
    df = df.drop(columns=["is_multi"])
    
    return df

cleaned_baseball = remove_duplicate_team_rows(cleaned_baseball)


# In[105]:


cleaned_baseball.dtypes


# **Which team employed the most amount of position players**
# - records with 2TM, 3TM and 4TM are removed for this analysis
# - smaller markets like tampa bay, minnesota and cleveland are towards the top as they don't have the funds to keep players when their contracts are up

# In[106]:


players_without_multiple = cleaned_baseball[~cleaned_baseball["Team"].isin(multiple)]


# In[107]:


order = players_without_multiple["Team"].value_counts().index
plt.figure(figsize=(16, 5))
sns.countplot(data=players_without_multiple, x="Team", order=order, hue="Team")
plt.title("Count of Unique Number of Position Players Employed")
plt.xlabel("Team Name")
plt.ylabel("Unique Count of Players Employed")
plt.show()


# In[108]:


order1 = cleaned_baseball["Age"].value_counts().index
plt.figure(figsize=(7, 3))
plt.title("Histogram of Players By Their Age")
plt.hist(data=cleaned_baseball, x="Age", bins=15, edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Unique Count of Players")
plt.show()


# In[109]:


homers = pd.pivot_table(cleaned_baseball, index="Player", values="HR", aggfunc="sum")
top_1o_pivot = homers.nlargest(10, "HR")
print(top_1o_pivot)


# In[110]:


strikeouts = pd.pivot_table(cleaned_baseball, index="Player", values="SO", aggfunc="sum")
most_strikeout_prone = strikeouts.nlargest(10, "SO")
print(most_strikeout_prone)


# In[111]:


allstar = cleaned_baseball[cleaned_baseball["All Star Appearance"] == True]

allstarappearance = pd.pivot_table(allstar, index="Player", values="All Star Appearance", aggfunc="sum")
most_all_stars = allstarappearance.nlargest(10, "All Star Appearance")
print(most_all_stars)


# In[112]:


MVP = cleaned_baseball[cleaned_baseball["MVP"] == True]

MVPWins = pd.pivot_table(MVP, index="Player", values="MVP", aggfunc="sum")
most_mvps = MVPWins.nlargest(10, "MVP")
print(most_mvps)


# In[79]:


home_runs_totals = cleaned_baseball.groupby("Year")["HR"].sum()

home_runs_totals.plot(kind="line", marker="o", color="blue", linewidth=2)
plt.title("Trend of Home Runs By Season")
plt.xlabel("Year")
plt.ylabel("Total Number of Home Runs")
plt.grid(True)
plt.show()


# In[80]:


sandiegopadres = cleaned_baseball[cleaned_baseball["Team"] == "SDP"]
padres_homers = sandiegopadres.groupby("Year")["HR"].sum()

padres_homers.plot(kind="line", marker="o", color="brown", linewidth=2)
plt.title("Trend of Home Runs By Season (San Diego Padres)")
plt.xlabel("Year")
plt.ylabel("Total Home Runs")
plt.grid(True)
plt.show()


# In[82]:


mannyandbryce = cleaned_baseball[cleaned_baseball["Player"].isin(["Manny Machado", "Bryce Harper"])]

home_runs = mannyandbryce.groupby(["Player", "Year"])["HR"].sum().unstack("Player")

home_runs.plot(kind="line", marker="o", linewidth=2)
plt.title("Trend og Home Runs By Season: Bryce Harper vs Manny Machado")
plt.xlabel("Year")
plt.ylabel("Total Home Runs")
plt.legend(title="Player")
plt.grid(True)
plt.show()


# In[ ]:




