"""
TP1 - Comprehension et visualisation des donnees

Membres (IFT799-52) :
  Hidaya LAKHDER - lakh5345
  Adam GUERROUANI - guea2637

Affichage des nuages de points (Methode 2 de l'enonce).
"""

import matplotlib
matplotlib.use("Agg")  # ecriture de fichiers sans fenetre graphique
import matplotlib.pyplot as plt


def nuage(X, y, classes, cols, titre, fichier):
    """Nuage de points de deux variables, colorie par classe."""
    plt.figure(figsize=(6, 5))
    for c in classes:
        plt.scatter(X[y == c, cols[0]], X[y == c, cols[1]], s=12, label=c)
    plt.xlabel(f"variable {cols[0]}")
    plt.ylabel(f"variable {cols[1]}")
    plt.title(titre)
    plt.legend()
    plt.savefig(fichier, dpi=150, bbox_inches="tight")
    plt.close()


def nuage_histogrammes(X, y, classes, cols, titre, fichier):
    """Nuage de points entoure de la distribution de chaque classe (affichage 1D).

    Les histogrammes en marge montrent la separation selon chaque variable
    prise isolement, le nuage central selon les deux ensemble.
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(6, 6),
        gridspec_kw={"width_ratios": [4, 1], "height_ratios": [1, 4]},
    )
    haut, centre, droite = axes[0, 0], axes[1, 0], axes[1, 1]
    axes[0, 1].axis("off")

    for c in classes:
        xc = X[y == c, cols[0]]
        yc = X[y == c, cols[1]]
        centre.scatter(xc, yc, s=12, label=c)
        haut.hist(xc, bins=25, alpha=0.5)
        droite.hist(yc, bins=25, alpha=0.5, orientation="horizontal")

    centre.set_xlabel(f"variable {cols[0]}")
    centre.set_ylabel(f"variable {cols[1]}")
    centre.legend()
    haut.set_xlim(centre.get_xlim())
    droite.set_ylim(centre.get_ylim())
    haut.set_xticks([])
    droite.set_yticks([])
    fig.suptitle(titre)
    fig.savefig(fichier, dpi=150, bbox_inches="tight")
    plt.close()
