#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 21:37:58 2025

@author: danyhourani
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#2: Preliminary steps

ds = pd.read_csv("exams.csv")  

print(ds.head()) # shows first 5 rows               
print("Shape:", ds.shape) # shows (rows,colums) the dataset has      
print("Info:")               
print(ds.info()) # show data types and if there are missing values
print("Describe:")
print(ds.describe()) # show basic statistics for numeric columns

print(ds.duplicated().sum())   # shows number of duplicates
ds = ds.drop_duplicates()      # removes them


print(ds.isnull().sum())  # checks how many missing values there are 

# Check current data types
print(ds.dtypes)

# Convert the three score columns to numeric
ds["math score"] = pd.to_numeric(ds["math score"])
ds["reading score"] = pd.to_numeric(ds["reading score"])
ds["writing score"] = pd.to_numeric(ds["writing score"])

#3: Univariate non-graphical EDA

# Numerical variables
print(ds[["math score", "reading score", "writing score"]].describe())  # mean, median(50%), std, quartiles
print("Variance:", ds[["math score", "reading score", "writing score"]].var())
print("Mode:", ds[["math score", "reading score", "writing score"]].mode().iloc[0])
print("Skewness:", ds[["math score", "reading score", "writing score"]].skew())
print("Kurtosis:", ds[["math score", "reading score", "writing score"]].kurt())

# Categorical variables
for all in ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]:
    print("Variable:", all)
    print("Frequency counts:")
    print(ds[all].value_counts())
    print("Proportions:")
    print(ds[all].value_counts(normalize=True).round(2))
    print("Mode:", ds[all].mode().iloc[0])
    print("Number of unique categories:", ds[all].nunique())

#4: Univariate graphical data

# Math Score

# a) Custom number of bins
plt.figure()
sns.histplot(ds["math score"], bins=20)
plt.title("Math score - histogram (bins=20)")
plt.xlabel("Math score")
plt.ylabel("Count")
plt.show()

# b) Conditioning on another variable
plt.figure()
sns.histplot(data=ds, x="math score", hue="gender", bins=20, element="step")
plt.title("Math score by gender - histogram")
plt.xlabel("Math score")
plt.ylabel("Count")
plt.show()

# c) Normalized histogram statistics
plt.figure()
sns.histplot(ds["math score"], bins=20, stat="probability")
plt.title("Math score - normalized histogram")
plt.xlabel("Math score")
plt.ylabel("Proportion")
plt.show()

# d) Kernel density estimation
plt.figure()
sns.kdeplot(ds["math score"], bw_adjust=1.0, fill=True)
plt.title("Math score - KDE (bw_adjust=1.0)")
plt.xlabel("Math score")
plt.ylabel("Density")
plt.show()


# Reading Score

# a) Custom number of bins
plt.figure()
sns.histplot(ds["reading score"], bins=20)
plt.title("Reading score - histogram (bins=20)")
plt.xlabel("Reading score")
plt.ylabel("Count")
plt.show()

# b) Conditioning on another variable
plt.figure()
sns.histplot(data=ds, x="reading score", hue="gender", bins=20, element="step")
plt.title("Reading score by gender - histogram")
plt.xlabel("Reading score")
plt.ylabel("Count")
plt.show()

# c) Normalized histogram statistics
plt.figure()
sns.histplot(ds["reading score"], bins=20, stat="probability")
plt.title("Reading score - normalized histogram")
plt.xlabel("Reading score")
plt.ylabel("Proportion")
plt.show()

# d) Kernel density estimation
plt.figure()
sns.kdeplot(ds["reading score"], bw_adjust=1.0, fill=True)
plt.title("Reading score - KDE (bw_adjust=1.0)")
plt.xlabel("Reading score")
plt.ylabel("Density")
plt.show()


# Writing Score

# a) Custom number of bins
plt.figure()
sns.histplot(ds["writing score"], bins=20)
plt.title("Writing score - histogram (bins=20)")
plt.xlabel("Writing score")
plt.ylabel("Count")
plt.show()

# b) Conditioning on another variable
plt.figure()
sns.histplot(data=ds, x="writing score", hue="gender", bins=20, element="step")
plt.title("Writing score by gender - histogram")
plt.xlabel("Writing score")
plt.ylabel("Count")
plt.show()

# c) Normalized histogram
plt.figure()
sns.histplot(ds["writing score"], bins=20, stat="probability")
plt.title("Writing score - normalized histogram")
plt.xlabel("Writing score")
plt.ylabel("Proportion")
plt.show()

# d) Kernel density estimation
plt.figure()
sns.kdeplot(ds["writing score"], bw_adjust=1.0, fill=True)
plt.title("Writing score - KDE (bw_adjust=1.0)")
plt.xlabel("Writing score")
plt.ylabel("Density")
plt.show()

#5: Multivariate non-graphical EDA

# a)

print("Counts: gender x lunch")
print(pd.crosstab(ds["gender"], ds["lunch"]))
print()

print("Counts: race/ethnicity x test prep")
print(pd.crosstab(ds["race/ethnicity"], ds["test preparation course"]))
print()

# b)

print("Proportion table: gender x lunch")
print(pd.crosstab(ds["gender"], ds["lunch"], normalize="index").round(2))
print()

print("Proportion table: race/ethnicity x test prep")
print(pd.crosstab(ds["race/ethnicity"], ds["test preparation course"], normalize="index").round(2))
print()

# c)

print("Three-way counts: [gender, test prep] x lunch")
print(pd.crosstab([ds["gender"], ds["test preparation course"]], ds["lunch"]))
print()


print("Three-way proportions: [gender, test prep] x lunch")
print(pd.crosstab([ds["gender"], ds["test preparation course"]], ds["lunch"], normalize="index").round(2))

#6: Multivariate graphical EDA

#6.1: Visualizing statistical relationships

# a) Faceting (using col parameter)
sns.relplot(data=ds, x="math score", y="reading score", kind="scatter", col="gender")
plt.show()

# b) Five variables at once (x, y, hue, size, col)
sns.relplot(data=ds, x="math score", y="writing score", hue="lunch", size="reading score", col="gender",kind="scatter")
plt.show()

# c) Linear regression
sns.lmplot(data=ds, x="reading score", y="writing score", hue="gender")
plt.show()

#6.2: Visualizing categorical data

# a) Categorical scatter (jitter disabled)
# I picked gender vs math score to see if guys or girls do better
sns.stripplot(data=ds, x="gender", y="math score", jitter=False)
plt.title("Math score by gender")
plt.xlabel("Gender")
plt.ylabel("Math score")
plt.show()

# b) Beeswarm plot (3 variables)
sns.swarmplot(data=ds, x="lunch", y="reading score", hue="gender")
plt.title("Reading score by lunch and gender")
plt.xlabel("Lunch")
plt.ylabel("Reading score")
plt.show()

# c) Boxen plot
sns.boxenplot(data=ds, x="race/ethnicity", y="math score")
plt.title("Math score by race/ethnicity")
plt.xlabel("Race/ethnicity")
plt.ylabel("Math score")
plt.show()

# d) Split violin plot (3 variables)
sns.violinplot(data=ds, x="lunch", y="writing score", hue="gender", split=True, bw=0.5, cut=0)
plt.title("Writing score by lunch and gender")
plt.xlabel("Lunch")
plt.ylabel("Writing score")
plt.show()

# e) Violin plot with scatter points inside
sns.violinplot(data=ds, x="gender", y="reading score", inner=None, cut=0)
sns.stripplot(data=ds, x="gender", y="reading score", dodge=False, alpha=0.4)
plt.title("Reading score by gender")
plt.xlabel("Gender")
plt.ylabel("Reading score")
plt.show()

# f) Point plot (3 variables, 90% CI, dashed lines)
sns.pointplot(data=ds, x="parental level of education", y="math score",
              hue="gender", ci=90, linestyles="--", dodge=True)
plt.title("Math score by parental education and gender")
plt.xlabel("Parental education")
plt.ylabel("Math score")
plt.show()

# g) Bar plot showing number of observations
sns.countplot(data=ds, x="race/ethnicity")
plt.title("Count by race/ethnicity")
plt.xlabel("Race/ethnicity")
plt.ylabel("Count")
plt.show()

#6.3: Visualizing bivariate distributions

# a) Heatmap plot
plt.figure()
sns.histplot( 
    data=ds,
    x="math score",
    y="reading score",
    bins=20,        # controls bin width (adjusted)
    cbar=True,      # adds color intensity bar
    cmap="Blues"    # color palette
)
plt.title("Heatmap of Math vs Reading scores")
plt.xlabel("Math score")
plt.ylabel("Reading score")
plt.show()

# b) Bivariate KDE (contour density plot)
sns.displot(
    data=ds,
    x="math score",
    y="writing score",
    kind="kde",        # kernel density estimation
    fill=True,         # fills contours
    levels=8,          # number of contour lines
    thresh=0.1,        # removes lowest-density outer layer
    cmap="mako"        # color style
)
plt.title("Bivariate KDE of Math vs Writing scores")
plt.xlabel("Math score")
plt.ylabel("Writing score")
plt.show()
 