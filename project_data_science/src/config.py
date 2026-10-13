"""
TP1 - Comprehension et visualisation des donnees

Membres (IFT799-52) :
  Hidaya LAKHDER - lakh5345
  Adam GUERROUANI - guea2637
"""

SEED = 42

N_GENES = 750         # < 801 patients : Sigma poolee reste inversible

CLASSES = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]

N_COMPOSANTES = 15    # ACP : "pas moins de 15"
PERPLEXITE = 30       # t-SNE
