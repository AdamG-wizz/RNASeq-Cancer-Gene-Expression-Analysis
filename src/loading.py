"""
TP1 - Comprehension et visualisation des donnees

Membres :
  Nom Prenom - CIP

Lecture du jeu de donnees, selection aleatoire des variables et mise en cache.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SEED, N_GENES

# Chemins relatifs a la racine du projet, pour que les fonctions marchent
# aussi bien depuis un notebook que depuis la racine.
RACINE = Path(__file__).resolve().parent.parent
DATA = RACINE / "data"
CACHE = RACINE / "cache"


def preparer():
    X = pd.read_csv(DATA / "data.csv", index_col=0)
    y = pd.read_csv(DATA / "labels.csv", index_col=0)["Class"]

    # Les genes de variance nulle ne separent rien et rendent Sigma singuliere.
    X = X.loc[:, X.var() > 0]

    rng = np.random.default_rng(SEED)
    genes = rng.choice(X.columns, N_GENES, replace=False)

    CACHE.mkdir(exist_ok=True)
    np.save(CACHE / "X.npy", X[genes].to_numpy())
    np.save(CACHE / "y.npy", y.to_numpy())
    np.savetxt(CACHE / "genes.txt", genes, fmt="%s")


def charger():
    X = np.load(CACHE / "X.npy")
    y = np.load(CACHE / "y.npy", allow_pickle=True)
    return X, y
