"""
TP1 - Comprehension et visualisation des donnees

Membres (IFT799-52) :
  Hidaya LAKHDER - lakh5345
  Adam GUERROUANI - guea2637

Execute l'ensemble du travail : exploration, Methode 1, Methode 2.
Lancer depuis ce dossier : python main.py

Sources : scikit-learn (PCA, TSNE, StandardScaler).
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.config import CLASSES, N_COMPOSANTES, PERPLEXITE, SEED
from src.distances import euclidienne, mahalanobis, sigma_inverse
from src.exploration import decrire, verifier_rang
from src.loading import preparer, charger
from src.plotting import nuage, nuage_histogrammes
from src.selection import selectionner
from src.separability import dist_intra, dist_inter, overlap


def titre(texte):
    print("\n" + "=" * 70)
    print(texte)
    print("=" * 70)


def exploration():
    """Decrit les donnees, constitue l'ensemble de depart et verifie Sigma."""
    titre("EXPLORATION DES DONNEES")
    decrire()

    print()
    preparer()
    X, y = charger()
    print("ensemble de depart :", X.shape)

    print()
    verifier_rang(X, y)
    return X, y


def methode1(X, y):
    """Overlap des 10 paires de classes, avec les deux metriques."""
    titre("METHODE 1 - SEPARABILITE SANS VISUALISATION")

    def tableau(dist):
        lignes = []
        for a, b in combinations(CLASSES, 2):
            Ca, Cb = X[y == a], X[y == b]
            lignes.append({
                "paire": f"{a}-{b}",
                "intra_1": dist_intra(Ca, dist),
                "intra_2": dist_intra(Cb, dist),
                "inter": dist_inter(Ca, Cb, dist),
                "overlap": overlap(Ca, Cb, dist),
            })
        t = pd.DataFrame(lignes).round(3)
        t["bien_separees"] = t["overlap"] < 1
        return t

    t_eucl = tableau(euclidienne)
    t_eucl.to_csv("results/method1_euclidienne.csv", index=False)
    print("\nDistance euclidienne :")
    print(t_eucl.to_string(index=False))

    # Covariance poolee : une seule Sigma estimee sur les 801 patients.
    t_maha = tableau(mahalanobis(sigma_inverse(X)))
    t_maha.to_csv("results/method1_mahalanobis.csv", index=False)
    print("\nDistance de Mahalanobis :")
    print(t_maha.to_string(index=False))

    comp = pd.DataFrame({
        "paire": t_eucl["paire"],
        "euclidienne": t_eucl["overlap"],
        "mahalanobis": t_maha["overlap"],
    })
    comp.to_csv("results/method1_comparaison.csv", index=False)
    print("\nComparaison des deux metriques :")
    print(comp.to_string(index=False))


def methode2_genes(X, y):
    """Selection exhaustive des meilleures paires de genes et affichage."""
    titre("METHODE 2 - VARIABLES ORIGINALES")

    n_paires = X.shape[1] * (X.shape[1] - 1) // 2
    print("evaluation des %d paires de genes..." % n_paires)

    paire_globale, meilleures = selectionner(X, y, CLASSES)
    pire = max(o for _, o in meilleures.values())
    print("meilleure paire globale : %s (pire Overlap %.3f)" % (paire_globale, pire))

    nuage(X, y, CLASSES, paire_globale,
          "Genes originaux - meilleure paire globale",
          "figures/original/globale.png")

    lignes = []
    for (a, b), (cols, o) in sorted(meilleures.items()):
        lignes.append({"paire": f"{a}-{b}", "variables": cols, "overlap": round(o, 3)})
        nuage(X, y, [a, b], cols,
              f"Genes originaux - {a} vs {b}",
              f"figures/original/{a}_{b}.png")
        # Partie optionnelle : distribution de chaque classe selon chaque variable.
        nuage_histogrammes(X, y, [a, b], cols,
                           f"{a} vs {b} (Overlap {o:.3f})",
                           f"figures/histogrammes/{a}_{b}.png")

    t = pd.DataFrame(lignes)
    t["bien_separees"] = t["overlap"] < 1
    t.to_csv("results/selection_originales.csv", index=False)
    print()
    print(t.to_string(index=False))


def methode2_acp(X, y):
    """ACP sur 15 composantes, puis meme selection que sur les genes."""
    titre("METHODE 2 - VARIABLES TRANSFORMEES PAR ACP")

    # Les ecarts-type des genes different fortement : on standardise.
    Z = StandardScaler().fit_transform(X)
    acp = PCA(n_components=N_COMPOSANTES, random_state=SEED)
    C = acp.fit_transform(Z)

    variance = pd.DataFrame({
        "composante": np.arange(1, N_COMPOSANTES + 1),
        "variance_expliquee": acp.explained_variance_ratio_.round(4),
        "variance_cumulee": np.cumsum(acp.explained_variance_ratio_).round(4),
    })
    variance.to_csv("results/variance_acp.csv", index=False)
    print(variance.to_string(index=False))
    print("\nvariance expliquee cumulee : %.3f" % acp.explained_variance_ratio_.sum())

    paire_globale, meilleures = selectionner(C, y, CLASSES)
    pire = max(o for _, o in meilleures.values())
    print("meilleure paire globale : CP%d, CP%d (pire Overlap %.3f)"
          % (paire_globale[0] + 1, paire_globale[1] + 1, pire))

    nuage(C, y, CLASSES, paire_globale,
          "ACP - meilleure paire globale",
          "figures/pca/globale.png")

    lignes = []
    for (a, b), (cols, o) in sorted(meilleures.items()):
        lignes.append({
            "paire": f"{a}-{b}",
            "composante_1": cols[0] + 1,
            "composante_2": cols[1] + 1,
            "overlap": round(o, 3),
        })
        nuage(C, y, [a, b], cols,
              f"ACP - {a} vs {b}",
              f"figures/pca/{a}_{b}.png")

    t = pd.DataFrame(lignes)
    t["bien_separees"] = t["overlap"] < 1
    t.to_csv("results/selection_acp.csv", index=False)
    print()
    print(t.to_string(index=False))


def methode2_tsne(X, y):
    """t-SNE : carte globale, une carte par paire de classes, perplexites."""
    titre("METHODE 2 - T-SNE")

    Z = StandardScaler().fit_transform(X)

    # t-SNE depend de l'initialisation : random_state est fixe.
    T = TSNE(n_components=2, perplexity=PERPLEXITE, random_state=SEED).fit_transform(Z)
    nuage(T, y, CLASSES, (0, 1),
          f"t-SNE - toutes les classes (perplexite {PERPLEXITE})",
          "figures/tsne/globale.png")

    lignes = []
    for a, b in combinations(CLASSES, 2):
        garde = (y == a) | (y == b)
        Tp = TSNE(n_components=2, perplexity=PERPLEXITE,
                  random_state=SEED).fit_transform(Z[garde])
        yp = y[garde]
        o = overlap(Tp[yp == a], Tp[yp == b], euclidienne)
        lignes.append({"paire": f"{a}-{b}", "overlap": round(o, 3)})
        nuage(Tp, yp, [a, b], (0, 1),
              f"t-SNE - {a} vs {b}",
              f"figures/tsne/{a}_{b}.png")

    t = pd.DataFrame(lignes)
    t["bien_separees"] = t["overlap"] < 1
    t.to_csv("results/selection_tsne.csv", index=False)
    print(t.to_string(index=False))

    # La diapositive 63 conseille de tester plusieurs perplexites.
    for p in [5, 50]:
        Tp = TSNE(n_components=2, perplexity=p, random_state=SEED).fit_transform(Z)
        nuage(Tp, y, CLASSES, (0, 1),
              f"t-SNE - perplexite {p}",
              f"figures/tsne/perplexite_{p}.png")


if __name__ == "__main__":
    X, y = exploration()
    methode1(X, y)
    methode2_genes(X, y)
    methode2_acp(X, y)
    methode2_tsne(X, y)
    titre("TERMINE")
