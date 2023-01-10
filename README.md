# LeNet5_Cuda

**TP Hardware for Signal Processing**

## Partie 1

Dans un premier temps, nous effectuons des opérations simples sur des matrices en utilisant à la fois le CPU et le GPU. Cette étude nous permet d'estimer la complexité des opérations, c'est-à-dire de déterminer combien de temps et de ressources sont nécessaires pour effectuer ces opérations sur les différents matériels.
Ensuite, on calcule l'accélération théorique obtenue en passant du CPU au GPU en comparant les performances de ces deux matériels sur les opérations de matrices. Cela nous permet de voir à quel point le GPU est plus performant que le CPU pour ce genre de tâche.
Enfin, on  mesure l'accélération réelle en comparant les temps d'exécution des opérations sur les matrices sur les différents matériels, cette mesure permettra de voir s'il y a un écart entre les résultats théoriques et les résultats pratiques.
Ces comparaisons nous permettent de voir quelles sont les opérations qui peuvent être accélérées de manière significative en utilisant un GPU, et comment cela peut impacter la performance globale des applications qui implémentent ces opérations.

Addition de matrice CPU
Le CPU nous permet seulement de traiter un processus à la fois donc pour l’addition d’une matrice de taille m x n la complexité de calcule sera de m x n . 

Addition de matrice GPU
Le GPU quant à lui, grâce à son architecture permet de paralléliser le traitement des processus. Pour une matrice de taille m x n il y a toujours m x n calcule à réaliser mais ces derniers sont réalisés en même temps.  

Multiplication  de matrice CPU
Pour la multiplication de deux matrice de taille m x n et n x p la complexité de calcule est de m x n x p 
Multiplication  de matrice GPU
La complexité de calcule est la même, avec les processus executé en parallèle donc ce sera significativement plus rapide

Nos tests ont été effectués sur une matrice 3000x3000. On relève les temps de calcul suivant : 

Time_ADD_CPU: 0.012000

Time_Mult_CPU: 106.648003

Time_ADD_GPU:0.003000

Time_Mult_GPU:0.003000

De cela on peut en déduire une accélération pour l’addition comme suit : 

Accélération = Temps d'exécution sans mécanisme d’accélération / Temps d'exécution avec mécanisme d’accélération 

Acc = Time_ADD_CPU/Time_ADD_GPU = 0.012000/0.003000 = 4

Acc = Time_Mult_CPU/Time_Mult_GPU = 106.648003/0.003000 = 35 549

Le cas de la multiplication présente une très forte accélération par rapport à l'addition, cela est dû au fait que la complexité de calcule est plus élevée pour cette tâche donc il y a un nombre de tâche parallélisable conséquent( cf. Amdahl’s Law : Accélération par rapport au nombre de threads exécutant la tâche).


## Partie 2
Nous avons implémenté la fonction de convolution 2D, la fonction maxpooling, et apres nous avons ensuite testé en utilisant la couche ci-dessous

Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST.

Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.

Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

 Afin de tester la convolution  on à modifier les valeurs de nos matrices raw_data et kernel afin qu'elles comprennent uniquement la valeurs 1 ainsi lors de la convolution le résultat obtenu est 6 matrices affichant la valeur 25, ce qui confirme le bon fonctionnement de notre fonction de convolution.
 
 Pour la fonction de sous échantillonnage(Maxpooling), il nous suffit de vérifier la taille de sortie de notre matrice 6x14x14 ainsi que les valeurs en son sain qui doivent toutes être 25.

 De plus il est important d’ajouter de la non-linéarité a notre CNN dans ce but on implémente la fonction d’activation tan qui est souvent utilisée dans les réseaux de neurones pour la reconnaissance d'images. Elle prend en entrée un nombre réel (M) et renvoie un autre nombre réel qui est compris entre -1 et 1.



## Partie 3

Nous avons implémenté LeNet-5 en CUDA

Couche de convolution 1: 6 filtres de convolution de taille 5x5, avec une stride de 1, avec une fonction d'activation de type Tanh.

Couche de sous-échantillonnage 1: pooling de taille 2x2 avec une stride de 2 (également appelé sous-échantillonnage par moyenne).

Couche de convolution 2: 16 filtres de convolution de taille 5x5, avec une stride de 1, avec une fonction d'activation de type Tanh.

Couche de sous-échantillonnage 2: pooling de taille 2x2 avec une stride de 2.

Couche de reconnaissance: 120 neurones fully-connected (ou pleinement connectés) avec une fonction d'activation de type Tanh.

Couche de reconnaissance 2: 84 neurones fully-connected avec une fonction d'activation de type Tanh.

Couche de sortie: 10 neurones fully-connected avec une fonction d'activation de softmax pour l'identification.

Et apres nous avons créer un réseau LeNet5 sur tensorflow puis l'entrainer sur le dataset MNIST. Une fois le modèle entraîné,nous enregistrons les poids de chaque couche dans un fichier .bin.

