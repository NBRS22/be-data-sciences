import numpy as np
import pandas as pd


# ============================
# 🔹 1. Lecture des datasets
# ============================

def read_ds(filename: str):
    
    with open(filename, "r", encoding="utf-8") as f:
        lignes = [line.strip().split(",") for line in f]
    max_cols = max(len(l) for l in lignes)
    for l in lignes:
        l += [np.nan] * (max_cols - len(l))
    return pd.DataFrame(lignes)

train_df = read_ds("train.csv")

test_df = read_ds("test.csv")