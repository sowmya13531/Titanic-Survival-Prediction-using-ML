from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
# Load Titanic dataset
titanic_data = sns.load_dataset("titanic").dropna(subset=["survived", "pclass", "sex"])
# Create a stratification column
titanic_data["stratify_col"] = titanic_data["survived"].astype(str) + "_" + \
                               titanic_data["pclass"].astype(str) + "_" + \
                               titanic_data["sex"].astype(str)
# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(titanic_data, titanic_data["stratify_col"]):
    strat_train_set = titanic_data.iloc[train_idx]
    strat_test_set = titanic_data.iloc[test_idx]
