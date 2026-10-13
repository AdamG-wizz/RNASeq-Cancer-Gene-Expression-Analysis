"""
TP1 - Comprehension et visualisation des donnees

Membres (IFT799-52) :
  Hidaya LAKHDER - lakh5345
  Adam GUERROUANI - guea2637

Selection automatique de paires de variables.

Strategie de recherche : toutes les paires de variables sont evaluees. La note 2
de l'enonce autorise une recherche partielle, mais le calcul vectorise reste
assez rapide pour rester exhaustif, ce qui garantit l'optimum.

Deux mesures d'evaluation :
  - pour une paire de classes donnee : l'Overlap de cette paire ;
  - pour toutes les paires de classes : le pire Overlap parmi les 10 paires,
    car une paire de variables n'est utile globalement que si elle separe
    aussi la paire de classes la plus difficile.
"""

from itertools import combinations

import numpy as np


def selectionner(X, y, classes):
    """Evalue toutes les paires de variables en un seul parcours.

    Retourne la meilleure paire pour toutes les classes, et un dictionnaire
    donnant la meilleure paire pour chaque paire de classes.
    """
    groupes = [X[y == c] for c in classes]
    centres = [G.mean(axis=0) for G in groupes]
    couples = list(combinations(range(len(classes)), 2))

    meilleure_globale, score_global = None, np.inf
    meilleures = {(classes[i], classes[j]): (None, np.inf) for i, j in couples}

    for cols in combinations(range(X.shape[1]), 2):
        c = list(cols)
        intra = [np.sqrt(((G[:, c] - m[c]) ** 2).sum(axis=1)).max()
                 for G, m in zip(groupes, centres)]

        pire = 0.0
        for i, j in couples:
            # Les points d'une classe sont mesures par rapport au centre de l'autre.
            di = np.sqrt(((groupes[i][:, c] - centres[j][c]) ** 2).sum(axis=1)).min()
            dj = np.sqrt(((groupes[j][:, c] - centres[i][c]) ** 2).sum(axis=1)).min()
            inter = min(di, dj)
            o = np.inf if inter == 0 else (intra[i] + intra[j]) / (2 * inter)

            paire = (classes[i], classes[j])
            if o < meilleures[paire][1]:
                meilleures[paire] = (cols, o)
            pire = max(pire, o)

        if pire < score_global:
            meilleure_globale, score_global = cols, pire

    return meilleure_globale, meilleures
