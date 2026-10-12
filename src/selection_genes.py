import numpy as np
import pandas as pd
from itertools import combinations


def selectionner_genes(X, y):
    classes = y.unique()
    donnees = X.to_numpy()
    etiquettes = y.to_numpy()

    # Patients de chaque classe
    patients = {}
    centroides = {}

    for classe in classes:
        patients[classe] = donnees[etiquettes == classe]
        centroides[classe] = patients[classe].mean(axis=0)

    # Meilleurs resultats
    meilleurs_globaux = []
    meilleures_paires = {}

    for classe1, classe2 in combinations(classes, 2):
        meilleures_paires[(classe1, classe2)] = {
            "Gene 1": None,
            "Gene 2": None,
            "Classe 1": classe1,
            "Classe 2": classe2,
            "Overlap": np.inf
        }

    # Evaluation de toutes les paires de genes
    for i, j in combinations(range(X.shape[1]), 2):
        gene1 = X.columns[i]
        gene2 = X.columns[j]

        distances_intra = {}

        for classe in classes:
            ecarts = patients[classe][:, [i, j]] - centroides[classe][[i, j]]
            distances = np.sqrt(np.sum(ecarts ** 2, axis=1))
            distances_intra[classe] = distances.max()

        scores = []

        for classe1, classe2 in combinations(classes, 2):
            centre1 = centroides[classe1][[i, j]]
            centre2 = centroides[classe2][[i, j]]

            ecarts1 = patients[classe1][:, [i, j]] - centre2
            ecarts2 = patients[classe2][:, [i, j]] - centre1

            d1 = np.sqrt(np.sum(ecarts1 ** 2, axis=1)).min()
            d2 = np.sqrt(np.sum(ecarts2 ** 2, axis=1)).min()

            inter = min(d1, d2)
            intra = distances_intra[classe1] + distances_intra[classe2]

            if inter == 0:
                overlap = np.inf
            else:
                overlap = intra / (2 * inter)

            scores.append(overlap)

            cle = (classe1, classe2)

            if overlap < meilleures_paires[cle]["Overlap"]:
                meilleures_paires[cle] = {
                    "Gene 1": gene1,
                    "Gene 2": gene2,
                    "Classe 1": classe1,
                    "Classe 2": classe2,
                    "Overlap": overlap
                }

        score_global = np.mean(scores)

        meilleurs_globaux.append({
            "Gene 1": gene1,
            "Gene 2": gene2,
            "Overlap moyen": score_global
        })

    globales = pd.DataFrame(meilleurs_globaux)
    globales = globales.sort_values("Overlap moyen").reset_index(drop=True)

    paires = pd.DataFrame(meilleures_paires.values())
    paires = paires.sort_values(["Classe 1", "Classe 2"]).reset_index(drop=True)

    return globales, paires
