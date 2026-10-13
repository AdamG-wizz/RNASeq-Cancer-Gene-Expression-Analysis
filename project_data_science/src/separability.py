"""
TP1 - Comprehension et visualisation des donnees

Membres (IFT799-52) :
  Hidaya LAKHDER - lakh5345
  Adam GUERROUANI - guea2637

Mesures de coherence et de separation (Methode 1).
"""

import numpy as np


def dist_intra(C, dist):
    """Distance maximale entre un point de C et le centre de C."""
    return dist(C, C.mean(axis=0)).max()


def dist_inter(C1, C2, dist):
    """Minimum des deux distances minimales croisees.

    Les points de C1 sont mesures par rapport au centre de C2, et
    inversement : les deux centres ne jouent pas le meme role.
    """
    a = dist(C1, C2.mean(axis=0)).min()
    b = dist(C2, C1.mean(axis=0)).min()
    return min(a, b)


def overlap(C1, C2, dist):
    """Indicateur de superposition. Overlap < 1 => classes bien separees."""
    inter = dist_inter(C1, C2, dist)
    if inter == 0:
        # Un patient se confond avec le centre de l'autre classe : separation nulle.
        # les classes ne satisfont pas le critère de bonne séparation
        return np.inf
    return (dist_intra(C1, dist) + dist_intra(C2, dist)) / (2 * inter)
