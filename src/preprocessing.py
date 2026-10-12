
import numpy as np
import pandas as pd

data = pd.read_csv("data/data.csv")
labels = pd.read_csv("data/labels.csv")

# Verification des patients
assert (data.iloc[:, 0] == labels.iloc[:, 0]).all()

# On garde les genes seulment
X = data.drop(columns=["Unnamed: 0"])
y = labels["Class"]

# 500 genes aleatoire
np.random.seed(42)
genes = np.random.choice(X.columns, 500, replace=False)
X = X[genes]

print(X.shape)
print(y.shape)
print(X.head())
