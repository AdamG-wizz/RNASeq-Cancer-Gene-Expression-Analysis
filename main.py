from src.preprocessing import X, y
from src.distances import calculer_overlaps
from src.selection_genes import selectionner_genes


euclidien = calculer_overlaps(X, y, "euclidienne")
euclidien.to_csv("results/tables/overlap_euclidien.csv", index=False)

print("Distances euclidiennes :")
print(euclidien)

mahalanobis = calculer_overlaps(X, y, "mahalanobis")
mahalanobis.to_csv("results/tables/overlap_mahalanobis.csv", index=False)

print("\nDistances de Mahalanobis :")
print(mahalanobis)

print("\nRecherche des meilleures paires de genes...")

globales, paires = selectionner_genes(X, y)

globales.to_csv("results/tables/meilleurs_genes_global.csv", index=False)
paires.to_csv("results/tables/meilleurs_genes_par_classe.csv", index=False)

print("\nMeilleures paires globales :")
print(globales.head(10))

print("\nMeilleures paires pour chaque combinaison de cancers :")
print(paires)
