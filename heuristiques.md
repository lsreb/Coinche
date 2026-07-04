# Ce fichier contient les heuristiques à suivre a priori

## 1) Enchères naïves

De façon générale, un joueur cherche à faire l'enchère maximale s'il a le choix entre plusieurs couleurs ou types d'enchère.

## 1.1) A la couleur

Si un joueur a 3 cartes d'une couleur dont un valet, il peut chercher un contrat dans cette couleur d'atout (parler à cette couleur) et parler à 80.
Si un joueur a 3 cartes d'une couleur dont le valet et le 9, il peut parler à cette couleur à 90.
Si un joueur a 4 cartes d'une couleur dont le valet et le 9, il peut parler à 100.



Pour annoncer directement  à 110 ou plus, le joueur doit au moins avoir le valet et 9 et 2 autres atouts, et il doit compter ses plis raisonnables : 1 pli par atout, puis en dehors des atouts:  1 pli par As, 1 pli par 10 troisième, 2 plis pour un 10 et roi troisième à la même couleur, 2 plis pour un AS et 10 à la même couleur. Le joueur annonce 110 s'il manque 3 plis, 120 s'il manque 2 plis, 130 s'il manque un pli, capot s'il pense avoir tous les plis.

Si un joueur parle à 80, son allié ne le remonte de +10 ( rajoute 10 au contrat) que s'il possède le 9 second minimum (donc le 9 d'atout et au moins un autre atout), ou s'il possède au moins 3 atouts. Dans ce cas l'équipe suppose qu'elle possède le valet et le 9 d'atout.
Si une équipe sait qu'elle a le valet et le 9, elle peut commencer à annonce les As hors atout ("exter") : on rajoute 10 points de plus au contrat par As exter en main, si la main contient au moins un atout. 


Si un adversaire a déjà parlé, le joueur doit "décrocher" : par exemple si l'adversaire a déjà parlé à 90, alors si le joueur parle à 100, cela signifie qu'il aurait voulu parler à 80 ou 90. S'il parle à 110, cela signifie qu'il voulait parler à 100 ou 110. Rien ne change pour 120 et plus dans ce cas.

Un décrochage jusqu'à 100, quand le joueur n'a pas la certitude que son partenaire a le 9 (donc sans confirmation de soutien d'atout), n'est possible que s'il compte lui-même au moins 4 plis dans sa propre main (atouts en main, sans compter automatiquement le 9, plus les plis hors atout comme un as exter). Ce décompte inclut alors déjà l'éventuel as exter : si le partenaire remonte ensuite l'enchère, ce joueur ne doit pas re-compter cet as exter comme une remontée supplémentaire, puisqu'il l'a déjà utilisé pour justifier son décrochage à 100.

Note sur l'enchère à couleur en réponse au partenaire qui a parlé à SA : on prend en compte les as du partenaire, -1 (potentiellement dans l'atout). Par exemple, si le partenaire a parlé à 90 SA et que l'adversaire avait dit 80 couleur, alors on comprend que le partenaire a deux as. On prend en compte un de ces as dans l'annonce à la couleur. Quand le partenaire remonte, il sait qu'un de ces as est déjà pris en compte, et potentiellement devra remonter le second s'il est hors atout, ou d'autres plis qu'il peut avoir.

# 1.2) Enchères SA

Un joueur peut parler à 80-SA avec 2 As et un 10 second en plus. Il peut parler à 90-SA s'il possède 3 As, et 100 SA s'il possède tous les As. 
Son partenaire remonte de 10 par As. 
Si une équipe a tous les As, elle rajoute 10 points par 10 qui n'est pas "sec" (donc isolé à la couleur dans une main), donc par 10 au moins second.

# 1.3) Enchères TA

Les enchères TA suivent le même principe que SA, en considérant les Valets plutôt que les As et les 9 plutôt que les 10.

# 2) Jeu naïf

Le système d'annonce et la stratégie est connue de tous les joueurs.

# 2.1) Jeu d'attaque

# 2.1.1) Jeu d'attaque à un contrat couleur

# 2.1.1.1) Jeu à l'atout

Dans un premier temps, si les adversaires ont encore des atouts a priori, l'attaque fait tomber les atouts en jouant à la couleur d'atout. Le joueur qui a initié le contrat (attaquant principal) met son valet au premier tour s'il l'a ; si c'est le partenaire suiveur du contrat qui ouvre, il joue son plus gros atout. Dans les deux cas, si le meneur n'a pas le valet et que son plus gros atout est le 9 (le 9 second, qu'il ait un seul autre atout ou plusieurs), il ne joue pas le 9 à ce tour mais le plus faible de ses autres atouts, pour garder le 9 pour le tour suivant.
S'il reste ensuite des atouts, l'attaquant principal joue son 9 d'atout s'il l'a, sinon son atout le plus faible. Le suiveur met son 9 d'atout s'il l'a, sinon son atout le plus faible.
S'il reste encore de l'atout chez l'adversaire, l'attaquant principal doit rejouer un petit atout pour le faire tomber.

# 2.1.1.2) Jeu hors atout

Si l'attaque a encore de l'atout : on applique les heuristiques de ce paragraphe. s'il n'y a plus d'atout, voir 2.1.2 le jeu à SA.
Chaque joueur cherche à devenir maitre du pli : les joueurs jouent leur carte maitre en priorité, comme l'As s'ils l'ont, ou le 10 si l'as est déjà passé dans un pli précédent. Sinon  ils jouent la plus petite carte possédée qui remporte le pli, et sinon la plus petite carte en main et sinon une défausse (voir section 2.3)
Donc si un joueur doit ouvrir à une couleur, soit il joue une carte maitre, soit une des plus petite carte possible.

En priorité, l'attaque veut ouvrir un pli dans ses couleurs où initialement il y a le plus de cartes. Si le partenaire a annoncé des as en plus lors des enchères, jusqu'au capot, l'attaque doit jouer dans les couleurs des as du partenaire.

# 2.1.2) Jeu d'attaque à SA

Ordre de priorité d'ouverture de couleur : 
-couleur longue (3 cartes ou plus) avec as : on ouvre de l'as, et plus tard de la pire carte sauf si on est encore maitre (si on peut jouer la meilleure carte restante)
-couleur longue sans as : on joue la plus petite carte
-couleur sans as : on joue la carte la plus forte

Style de jeu : sauf pour récupérer un 10 adverse, on évite de jouer son as au premier tour d'une couleur. On essaie de récupérer la maitrise des plis sans investir d'as au premier tour d'une couleur.
Si un as est déjà passé, et qu'on a le 10, on le joue pour devenir maitre, et on essaie de rejouer dans cette couleur.

Quand l'attaque n'a plus qu'une couleur forte en main (un seul As maitre, pas de 10 second), il ouvre avec son as.

# 2.1.3) Jeu d'attaque à TA

Voir 2.1.2 à SA, mais en considérant les valets au lieu des As, et les 9 au lieu des 10.

# 2.2) Jeu de défense

# 2.2.1) Jeu de défense à contrat couleur

Quand la défense coupe, elle veut jouer les plus gros possibles à la première coupe possible, sauf si elle a le 9 troisième à l'atout, ou l'as quatrième. Dans ce cas elle joue un petit atout à la place si nécessaire.


Si elle doit ouvrir le premier pli, elle ouvre dans un as hors atout si elle peut. Sinon elle joue dans une singlette (couleur avec une seule carte).
Si un joueur récupère la main après l'ouverture de son partenaire, il avait probablement une singlette. Ce joueur rejoue à cette couleur si possible.

En général, après le début de la donne, la défense cherche à jouer comme une défense SA, voir 2.2.2.

# 2.2.2) Jeu de défense à SA

Si le contrat est 80-SA, la défense joue comme à l'attaque SA, voir 2.2.1.
Sinon, la défense joue ses as à la première occasion.
Sinon on suit la règle du jeu hors-atout 2.1.1.2

# 2.2.3) Jeu de défense à TA

On suit les mêmes principes que 2.2.2 à SA, mais en considérant les valets et 9 à la place des as et 10 respectivement.

# 2.3) Défausse

Choix de la couleur à défausser : on défausse une couleur faible, c'est à dire une singlette initiale ou une couleur avec deux cartes initialement, dans les deux cas à condition que ces cartes ne soient ni le 10 ni l'as. Sinon une couleur aléatoire

Quand un joueur est amené à défausser :
- si son partenaire est maitre absolu (avec une carte maitre à l'atout ou à la couleur), on défausse la carte la plus forte de la couleur choisie, qui n'est pas une carte maitre absolue.
- sinon on défausse la pire carte de la couleur choisie.

# 2.4) Indications stratégiques générales

De façon générale, les joueurs doivent, pour marquer des plis et donc des points, avoir le plus de cartes maitres possibles. Pour compter des plis potentiels par exemples, et essayer de jouer au mieux, hors atout par exemple, il est donc important de compter les As, puis les 10 derrière les As, et à partir du 3e pli à une couleur, la probabilité qu'une carte quelconque devienne maitre, même un petit 7, augmente, car on peut être le dernier joueur à en avoir : c'est ce qu'on appelle avoir une longue. Lors du jeu, il convient donc, dans le cas général naif, de mettre ses as au premier tour d'une couleur, les 10 au second, et aviser ensuite avec les cartes maitres restantes de la couleur, et ce, pour chaque couleur. On procède de façon similaire avec les atouts, avec le valet et le 9, et c'est pour ça qu'on joue le valet au premier tour, et le 9 au second, quand deux partenaires ont ensemble le valet et le 9. 