import numpy as np
import pandas as pd
from itertools import combinations


def distance_euclidienne(x, y):
    return np.linalg.norm(x - y)


def distance_mahalanobis(x, y, inverse_cov):
    difference = x - y
    return np.sqrt(max(0, difference @ inverse_cov @ difference))


def calculer_centroides(X, y):
    centroides = {}

    for classe in y.unique():
        centroides[classe] = X[y == classe].mean().to_numpy()

    return centroides


def calculer_overlaps(X, y, metrique="euclidienne"):
    centroides = calculer_centroides(X, y)

    if metrique == "mahalanobis":
        covariance = np.cov(X.to_numpy(), rowvar=False)
        inverse_cov = np.linalg.pinv(covariance)

        def distance(x, centre):
            return distance_mahalanobis(x, centre, inverse_cov)

    elif metrique == "euclidienne":
        distance = distance_euclidienne

    else:
        raise ValueError("Metrique inconnue")

    distances_intra = {}

    for classe in y.unique():
        patients = X[y == classe].to_numpy()
        centre = centroides[classe]

        distances_intra[classe] = max(
            distance(patient, centre) for patient in patients
        )

    resultats = []

    for classe1, classe2 in combinations(y.unique(), 2):
        patients1 = X[y == classe1].to_numpy()
        patients2 = X[y == classe2].to_numpy()

        d1 = min(
            distance(patient, centroides[classe2])
            for patient in patients1
        )

        d2 = min(
            distance(patient, centroides[classe1])
            for patient in patients2
        )

        inter = min(d1, d2)

        if inter == 0:
            overlap = np.inf if distances_intra[classe1] + distances_intra[classe2] > 0 else np.nan
        else:
            overlap = (
                distances_intra[classe1] + distances_intra[classe2]
            ) / (2 * inter)

        resultats.append({
            "Classe 1": classe1,
            "Classe 2": classe2,
            "Intra 1": distances_intra[classe1],
            "Intra 2": distances_intra[classe2],
            "Inter": inter,
            "Overlap": overlap
        })

    return pd.DataFrame(resultats)
