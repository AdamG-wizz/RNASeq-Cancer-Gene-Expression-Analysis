"""
TP1 - Comprehension et visualisation des donnees

Membres (IFT799-52) :
  Hidaya LAKHDER - lakh5345
  Adam GUERROUANI - guea2637

Mesures de distance.
"""

import numpy as np


def euclidienne(P, q):
    """Distance entre chaque ligne de P et le point q."""
    return np.sqrt(((P - q) ** 2).sum(axis=1))


def mahalanobis(Sigma_inv):
    """Retourne une fonction de distance de Mahalanobis pour une Sigma donnee.

    Sigma est commune sur l'ensemble des patients.
    """
    def distance(P, q):
        d = P - q
        return np.sqrt((d @ Sigma_inv * d).sum(axis=1))
    return distance


def sigma_inverse(X):
    return np.linalg.inv(np.cov(X, rowvar=False))
