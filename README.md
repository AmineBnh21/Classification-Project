Partie 1 — Le dataset

Fichier : data/airportsAndCoordAndPop.graphml

Le graphe représente le réseau mondial des lignes aériennes :

Nœud = une ville desservie par un aéroport
Arête = il existe une ligne aérienne directe entre deux villes (graphe non orienté, sans poids)
Attributs des nœuds
Attribut	Type	Description
lat	float	Latitude
lon	float	Longitude
population	int	Population de la ville
country	str	Pays (cible de T1)
city_name	str	Nom de la ville (non utilisé comme feature)
Statistiques principales
Mesure	Valeur
Nombre de nœuds	3 363
Nombre d'arêtes	13 547
Composantes connexes	6 (dont une géante de 3 351 nœuds)
Degré moyen / médian / max	8,06 / 3 / 248
Plus gros hubs	Paris (248), Londres (240), Francfort (234), Amsterdam (191), Chicago (183)
Nombre de pays	212
Pays avec une seule ville	62
Homophilie des arêtes (même pays)	0,57
Corrélation degré ↔ log(population)	0,36
⚠️ Trois pièges identifiés lors de l'exploration

Piège 1 — Les coordonnées contiennent presque la réponse. Un simple 1-NN sur (lat, lon), sans aucune arête, retrouve le bon pays dans ≈ 85 % des cas (50 % des nœuds en test). Les villes d'un même pays sont regroupées géographiquement. Si on fournit les coordonnées au GNN, on ne peut plus mesurer l'apport du graphe. → Solution : évaluer tous les modèles dans deux scénarios, avec et sans coordonnées.

Piège 2 — La moitié des populations sont une valeur par défaut. 1 673 villes (≈ 50 %) ont une population d'exactement 10 000. C'est une valeur de remplissage pour « inconnu », pas une mesure réelle. → Solution : ces nœuds restent dans le graphe (ils transmettent des messages) mais sont exclus de la loss et de l'évaluation pour T2.

Piège 3 — Les labels de pays sont bruités et déséquilibrés. Certains noms contiennent de l'encodage HTML (M&Eacute;XICO#MEXICO, PER&Uacute;#PERU, VI&Ecirc;T_NAM#VIET_NAM,VIETNAM…). Les classes sont très déséquilibrées (USA : 650 villes, 62 pays avec une seule ville). → Solution : nettoyage des noms, regroupement des pays rares dans une classe OTHER, et utilisation du F1-macro en plus de l'accuracy.

Partie 2 — Nettoyage et préparation des données

Script : src/data.py

2.1 Nettoyage des labels de pays
Si le nom contient #, garder la partie après le # (ex. M&Eacute;XICO#MEXICO → MEXICO).
Si le résultat contient une virgule, garder la première forme (ex. VIET_NAM,VIETNAM → VIET_NAM).
Regrouper les pays ayant moins de 5 villes dans la classe OTHER.
2.2 Cible de population
y_pop = log10(population)
mask_pop = (population != 10000) → seuls ces nœuds servent à entraîner et évaluer T2.
2.3 Features des nœuds
Groupe	Features	Scénario A (avec coords)	Scénario B (sans coords)
Géographiques	lat, lon normalisées, ou encodage sphérique (x, y, z)	✅	❌
Structurelles	degré (log), PageRank, clustering, centralité de proximité	✅	✅
Constante	un vecteur de 1 (si aucune autre feature)	—	—

Note : pour T1, la population n'est pas utilisée comme feature (et inversement le pays n'est jamais utilisé comme feature pour T2).

Astuce : l'encodage (x, y, z) = (cos lat · cos lon, cos lat · sin lon, sin lat) évite la discontinuité entre −180° et +180° de longitude (le Pacifique).

2.4 Conversion

Le graphe NetworkX est converti en objet torch_geometric.data.Data contenant x, edge_index, y_country, y_pop, mask_pop, puis sauvegardé dans data/processed/.

Partie 3 — Protocole expérimental

Script : src/splits.py

3.1 Scénarios évalués
Scénario	Description
A — avec coordonnées	Toutes les features disponibles
B — sans coordonnées	Features structurelles uniquement : on mesure l'apport propre du graphe
Taux de labels	10 %, 30 %, 50 % des nœuds labellisés pour l'entraînement
3.2 Découpage
Pour chaque taux de labels : train = X %, val = 10 %, test = le reste.
Découpage stratifié par pays pour T1 (dans la mesure du possible).
Le graphe complet est toujours visible (cadre transductif) : seuls les labels sont cachés.
3.3 Robustesse
Chaque expérience est répétée sur 10 graines : 0, 1, …, 9.
Les résultats sont donnés en moyenne ± écart-type.
Sélection du meilleur epoch par early stopping sur la validation (patience = 50).
3.4 Hyperparamètres

Recherche avec Optuna sur l'ensemble de validation uniquement (jamais sur le test) :

Hyperparamètre	Plage
Dimension cachée	32, 64, 128, 256
Nombre de couches	1 à 4
Dropout	0,0 à 0,6
Learning rate	1e-4 à 1e-2 (log)
Weight decay	0 à 5e-4
Têtes d'attention (GAT)	1, 2, 4, 8
Partie 4 — Baselines

Script : src/baselines.py

Ces modèles servent de points de comparaison. Ils permettent de savoir si les GNN apportent réellement quelque chose.

Baseline	Utilise le graphe ?	Utilise les features ?	T1	T2
Classe majoritaire / moyenne	❌	❌	✅	✅
MLP	❌	✅	✅	✅
k-NN sur coordonnées	❌	✅ (coords)	✅ (scénario A)	✅ (scénario A)
Label Propagation	✅	❌	✅	❌
Node2Vec + régression logistique / linéaire	✅	❌ (embeddings)	✅	✅

Ce qu'on attend :

Scénario A : le k-NN est une baseline forte (≈ 85 %). Un GNN doit la battre pour être utile.
Scénario B : le MLP devrait s'effondrer, tandis que la Label Propagation et Node2Vec devraient bien résister.