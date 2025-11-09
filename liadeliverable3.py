#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 21:37:58 2025

@author: danyhourani
"""

import pandas as pd

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



