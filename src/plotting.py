"""
IFT599-1 / IFT799-51 / IFT799-52
TP1 - Comprehension et visualisation des donnees

Membres :
  Nom Prenom - CIP

Affichage des nuages de points (Methode 2 de l'enonce).
"""

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
    plt.show()
