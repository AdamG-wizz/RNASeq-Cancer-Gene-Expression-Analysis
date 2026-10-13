"""
TP1 - Comprehension et visualisation des donnees

Membres (IFT799-52) :
  Hidaya LAKHDER - lakh5345
  Adam GUERROUANI - guea2637

Exploration du jeu de donnees : verifications avant traitement.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from src.config import CLASSES
from src.loading import DATA


def decrire():
    """Decrit le jeu de donnees et verifie s'il necessite un nettoyage."""
    X = pd.read_csv(DATA / "data.csv", index_col=0)
    y = pd.read_csv(DATA / "labels.csv", index_col=0)["Class"]

    print("dimensions :", X.shape)
    print(y.value_counts().to_string())

    print()
    print("valeurs manquantes :", X.isna().sum().sum())
    print("patients en double :", X.index.duplicated().sum())
    print("patients alignes avec les etiquettes :", (X.index == y.index).all())
    print("intervalle des valeurs : %.2f a %.2f" % (X.values.min(), X.values.max()))
    print("proportion de zeros : %.1f%%" % (100 * (X.values == 0).mean()))

    nulles = (X.var() == 0).sum()
    print()
    print("genes de variance nulle :", nulles)
    print("genes restants :", X.shape[1] - nulles)

    # Echelle et correlation : justifient la standardisation avant l'ACP
    # et l'interet de la distance de Mahalanobis.
    Xr = X.loc[:, X.var() > 0]
    s = Xr.std()
    print()
    print("ecarts-type : min %.2f  max %.2f  (rapport x%.0f)"
          % (s.min(), s.max(), s.max() / s.min()))

    ech = Xr.sample(500, axis=1, random_state=0)
    C = np.corrcoef(ech, rowvar=False)
    hors_diag = C[~np.eye(500, dtype=bool)]
    print("correlation moyenne (valeur absolue) : %.3f" % np.abs(hors_diag).mean())
    print("paires de genes avec |r| > 0.5 : %d" % ((np.abs(hors_diag) > 0.5).sum() // 2))


def verifier_rang(X, y):
    """Verifie que la covariance poolee est inversible.

    Une covariance estimee classe par classe serait singuliere : son rang ne
    peut depasser le nombre de patients de la classe moins un.
    """
    S = np.cov(X, rowvar=False)
    print("Sigma poolee : rang %d sur %d" % (np.linalg.matrix_rank(S), S.shape[0]))

    for c in CLASSES:
        Xc = X[y == c]
        Sc = np.cov(Xc, rowvar=False)
        print("  %s (%3d patients) : rang %3d sur %d"
              % (c, len(Xc), np.linalg.matrix_rank(Sc), Sc.shape[0]))
