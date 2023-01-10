# LeNet5_Cuda

**TP Hardware for Signal Processing**

## Partie 1

**Dans un premier temps, nous effectuons des opérations simples sur des matrices en utilisant à la fois le CPU et le GPU. Cette étude nous permet d'estimer la complexité des opérations, c'est-à-dire de déterminer combien de temps et de ressources sont nécessaires pour effectuer ces opérations sur les différents matériels.
Ensuite, on calcule l'accélération théorique obtenue en passant du CPU au GPU en comparant les performances de ces deux matériels sur les opérations de matrices. Cela nous permet de voir à quel point le GPU est plus performant que le CPU pour ce genre de tâche.
Enfin, on  mesure l'accélération réelle en comparant les temps d'exécution des opérations sur les matrices sur les différents matériels, cette mesure permettra de voir s'il y a un écart entre les résultats théoriques et les résultats pratiques.
Ces comparaisons nous permettent de voir quelles sont les opérations qui peuvent être accélérées de manière significative en utilisant un GPU, et comment cela peut impacter la performance globale des applications qui implémentent ces opérations.**

**Addition de matrice CPU
Addition de matrice GPU
Multiplication  de matrice CPU
Multiplication  de matrice GPU**

## Partie 2

Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST
Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.
Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

## Partie 3
