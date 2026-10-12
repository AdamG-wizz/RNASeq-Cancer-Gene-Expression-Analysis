
import pandas as pd

data = pd.read_csv("data/data.csv")
labels = pd.read_csv("data/labels.csv")

print(data.shape)
print(labels.shape)

print(data.head())
print(labels.head())

print(data.columns[:10])
print(labels.columns)

print(data.isnull().sum().sum())
print(labels.isnull().sum().sum())

print(labels["Class"].value_counts())
