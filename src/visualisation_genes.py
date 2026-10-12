
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import X, y


os.makedirs("results/figures/genes", exist_ok=True)

sns.set_theme(style="whitegrid")


def tracer_genes(gene1, gene2, classes, titre, fichier):
    plt.figure(figsize=(9, 6))

    for classe in classes:
        patients = X[y == classe]

        plt.scatter(
            patients[gene1],
            patients[gene2],
            label=classe,
            alpha=0.65,
            s=22
        )

    plt.xlabel(gene1)
    plt.ylabel(gene2)
    plt.title(titre)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fichier, dpi=300)
    plt.close()


def visualiser_genes():
    globales = pd.read_csv("results/tables/meilleurs_genes_global.csv")
    paires = pd.read_csv("results/tables/meilleurs_genes_par_classe.csv")

    # Graphique global : les cinq cancers
    gene1 = globales.iloc[0]["Gene 1"]
    gene2 = globales.iloc[0]["Gene 2"]

    tracer_genes(
        gene1,
        gene2,
        y.unique(),
        "Visualisation des cinq cancers - genes originaux",
        "results/figures/genes/global.png"
    )

    # Un graphique pour chaque paire de cancers
    for _, ligne in paires.iterrows():
        classe1 = ligne["Classe 1"]
        classe2 = ligne["Classe 2"]

        tracer_genes(
            ligne["Gene 1"],
            ligne["Gene 2"],
            [classe1, classe2],
            f"{classe1} vs {classe2} - genes originaux",
            f"results/figures/genes/{classe1}_{classe2}.png"
        )

    print("11 graphiques enregistres dans results/figures/genes/")


if __name__ == "__main__":
    visualiser_genes()
