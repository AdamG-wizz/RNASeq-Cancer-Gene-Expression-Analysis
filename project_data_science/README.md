# TP1 - Comprehension et visualisation des donnees


Membres (IFT799-52) :

- Hidaya LAKHDER - lakh5345
- Adam GUERROUANI - guea2637

## Execution

```
python main.py
```

Duree : environ 10 minutes. La recherche exhaustive des paires de genes
(280 875 combinaisons) represente l'essentiel de ce temps.

Les fichiers `data/data.csv` et `data/labels.csv` doivent etre presents.

## Organisation

```
main.py                 enchaine les cinq etapes
src/
  config.py             constantes (seed, nombre de genes, perplexite)
  loading.py            lecture, filtrage, tirage aleatoire, cache
  exploration.py        verifications sur les donnees brutes
  distances.py          distances euclidienne et de Mahalanobis
  separability.py       distances intra, inter et Overlap
  selection.py          recherche des meilleures paires de variables
  plotting.py           nuages de points et histogrammes
cache/                  ensemble de depart (genes.txt = variables retenues)
results/                tableaux au format csv
figures/                nuages de points et histogrammes
```

## Etapes

1. **Exploration** : description des donnees, verification des valeurs
   manquantes et de l'alignement des fichiers, retrait des genes de variance
   nulle, tirage de l'ensemble de depart, verification du rang de la covariance.
2. **Methode 1** : Overlap des dix paires de classes, avec la distance
   euclidienne puis celle de Mahalanobis, sur toutes les variables.
3. **Methode 2 - variables originales** : recherche exhaustive des meilleures
   paires de genes, nuages de points et histogrammes (partie optionnelle).
4. **Methode 2 - ACP** : 15 composantes principales, meme selection.
5. **Methode 2 - t-SNE** : carte globale, une carte par paire de classes, et
   comparaison de trois perplexites.

## Ensemble de depart

Les 20531 genes sont reduits en deux temps :

1. retrait des 267 genes de variance nulle, qui ne contribuent a aucune distance
   et rendent la matrice de covariance singuliere ;
2. tirage aleatoire de 750 genes parmi les 20264 restants (seed 42).

La liste des genes retenus est enregistree dans `cache/genes.txt`.

Le nombre de 750 est impose par la distance de Mahalanobis : la covariance est
estimee sur 801 patients, son rang ne peut donc depasser 800. Au-dela, la
matrice devient singuliere. A 750 variables elle reste de rang plein et bien
conditionnee.

La covariance est poolee, c'est-a-dire estimee une seule fois sur les 801
patients. Une covariance par classe serait singuliere, chaque classe comptant
moins de 750 patients. Les deux metriques portent ainsi sur les memes donnees.

## Sources

- scikit-learn : `PCA`, `TSNE`, `StandardScaler`
- Les mesures de separabilite (intra, inter, Overlap) et la selection des paires
  de variables sont implantees a partir des formules de l'enonce.
