Ce fichier rappelle les règles de la belote coinchée, appelée coinche.
# Spécifications Techniques : Belote Coinchée pour l'Apprentissage par Renforcement (RL)

## 1) Contexte
La coinche est un jeu de cartes à jouer qui se joue avec les 32 cartes des 4 couleurs, avec l'as (A), le roi (K), la dame (Q), le valet (J), le 10, le 9, le 8 et le 7. Elle se joue à 4 joueurs (nord, ouest, sud et est), deux équipes de deux joueurs, nord avec sud et est avec ouest. C'est un jeu à plis, il y a donc 8 plis. A la fin d'un pli, un joueur remporte le pli, et pourra initier le pli suivant. Quand les 8 plis sont terminés, on compte les points totaux dans chaque pli, et on compare au contrat pour déterminer le vainqueur du contrat. Les cartes en main sont cachées aux autres joueurs.

Une partie de coinche est composée de plusieurs donnes. Une donne se déroule de la façon suivante : le donneur coupe le paquet, distribue 8 cartes à chaque joueur (3 à chacun, 3 à chacun, puis 2 à chacun). Ensuite en commençant par la droite du donneur, les joueurs font leurs enchères. Quand les 4 joueurs ont passé, les enchères sont terminées et le contrat est fixé. On joue ensuite la donne, donc les 8 plis, et on determine le vainqueur du contrat. Pour la donne suivante, le donneur devient le joueur à droite du donneur précédent.

## 2) Déroulement  précis de la phase d'enchères

Le joueur à la droite du donneur commence la phase d'enchères. Il annonce une dizaine entre 80 et 160 inclus, ou capot, ou il passe son tour. Il annonce aussi une couleur comme pique (P) coeur (C) carreau (K) ou trèfle (T), ou "sans-atout" (SA) ou "tout-atouts" (TA). Il dit par exemple 80C. Le joueur à sa droite doit passer ou surenchérir, donc avec au moins 10 de plus, dans n'importe quelle couleur, SA ou TA, ou alors coincher. Quand un joueur coinche, le contrat est fixé à la dernière annonce faite, et les mises sont doublées. Quand trois joueurs passent à la suite, le quatrième peut surenchérir sur lui-même, mais seulement s'il change de couleur, de TA ou SA. Quand les 4 joueurs passent, le contrat est fixé. L'équipe qui a pris le contrat est dite "équipe à l'attaque" et l'autre "équipe en défense".
Si aucun joueur n'a fait d'enchères, la donne est nulle, personne ne marque de point, on passe à la donne suivante en rebattant les cartes avec le nouveau donner qui est le prochain joueur dans le sens anti-horaire.

## 3) Ordre des cartes

A l'atout : J-9-A-10-K-Q-8-7, les valeurs de points respectives sont 20-14-11-10-4-3-0-0.
Aux couleurs normalees : A-10-K-Q-J-9-8-7, avec comme valeurs de points respectives 11-10-4-3-2-0-0-0.
A sans-atout (SA), l'ordre est le même qu'aux couleurs normales, mais les valeurs de points respectives sont : 19-10-4-3-2-0-0-0.
A tout-atout (TA), l'ordre et les valeurs de toutes les couleurs sont celles de l'atout.
Quand on joue à la couleur, les atouts sont plus forts que les trois autres couleurs. 
Quand on joueur à la couleur, si un joueur quelconque, à l'attaque ou la défense possède la dame et le roi d'atouts, au moment de poser l'un ou l'autre, il annonce verbalement "belote". Il marquera alors 20 points supplémentaires lors du comptage, et ces points peuvent servir à effectuer le contrat à l'attaque.



## 4) Déroulement du jeu et des plis

Le joueur à la droite du donneur commence le premier pli (il ouvre) avec une carte de sa main, qui va déterminer la couleur demandée au pli.

Quand un joueur ouvre un pli, en jouant dans le sens anti-horaire (N > W > S > E > N), les joueurs doivent si possible joueur la carte de la même couleur que le premier joueur.

On se place d'abord dans le cas où un contrat a été pris à une couleur simple. Par exemple, si pique est la couleur d'atout, et que le premier joueur ouvre à coeur, tout le monde doit fournir du coeur. Un joueur qui ne peut pas fournir du coeur doit a priori couper, donc mettre de l'atout, s'il en a. Il y a plusieurs cas particuliers ici : si son partenaire est maitre du pli, c'est à dire qu'il remporterait le pli dans l'état actuel, alors le joueur peut se défausser au lieu de couper, donc jouer n'importe quelle carte. Si un adversaire a déjà coupé, et que le joueur a de l'atout, il doit surcouper (couper avec une carte de valeur supérieure), et s'il a de l'atout mais seulement inférieur, il peut le conserver, et se défausser. Dans les autres cas, un joueur qui n'a ni la couleur demandée ni de l'atout doit se défausser.
Par exemple : l'atout est pique. Nord ouvre avec A-C (as de coeur), ouest n'a pas de pique et coupe avec 10-P (10 de pique), sud n'a pas de coeur, et à l'atout il a l'as d'atout et le 8 d'atout, il doit donc jouer l'as d'atout, à pique (A-P). Est finalement n'a qu'un atout, le roi de pique K-P, mais il ne bat pas l'atout, et peut donc se défausser, par exemple de son 7 de trèfle (7-T).
Ensuite le joueur remportant le pli, suivant l'ordre indiqué en section 3, ouvrira le suivant.
Si un joueur ouvre à l'atout, les autres joueurs doivent si possible jouer à atout, en montant en force si possible. S'ils ne peuvent pas, il jouent n'importe quel atout inférieur. S'ils n'ont plus d'atout, ils doivent défausser à n'importe quelle couleur.

## 5) Comptage des points

A la fin des 8 plis, on compte les points de chaque plis remporté par l'attaque et la défense séparément. Si le contrat est à couleur ou SA, l'équipe qui remporte le dernier pli gagne 10 points bonus, appelé le "10 de der" ou "la der". Il n'y a pas de der à TA. A la couleur ou SA, il y a donc 162 points en tout. L'attaque remplit son contrat si elle a marqué plus de points que le nombre choisi dans son contrat. Par exemple si elle marque 75 points avec ses plis et en comptant la der, et son contrat était 80-C, alors le contrat n'est pas rempli.
A la couleur, si un même joueur joue le roi et la dame d'atouts, en posant un des deux il peut annoncer verbalement "belote". Son équipe marquera alors 20 points bonus à la fin de la donne, qui peuvent permettre de faire le contrat. Si l'équipe marquait 75 points, avec la belote elle passe à 95 et pourrait faire son contrat à 80-C, dans l'exemple précédent.
Pour faire capot, une équipe doit faire tous les 8 plis de la donne : le capot est un contrat à 250.

Si une équipe remporte son contrat, si c'était capot annoncé, elle marque 500 et les 20 points d'une potentielle belote, la défense marque 0. Si elle remporte un autre type de contrat, elle remporte les points annoncés du contrat plus les points faits, en arrondissant les unités 5,6, 7, 8 et 9 à la dizaine supérieure, et les unités 1, 2, 3 et 4 à la dizaine inférieure. La défense marque ses points faits, arrondis aussi de la même façon et avec une belote si elle l'a. Si une équipe fait tous les plis, les points faits valent 250 et pas 160.
Par exemple, pour un contrat à 90-T, l'attaque a fait 107, la défense 55 et une belote. L'attaque marque 90+110 donc 200 points, et la défense 60+20 donc 80 points. 
Ces points s'additionnent d'une donne à l'autre, se cumulent, et la première équipe à 3000 gagne.
En cas de contrat chuté, l'attaque marque 0 point (plus 20 potentiels de sa belote), la défense marque 160+ la valeur du contrat, plus une potentielle belote.
En cas de coinche annoncée, la mise est doublée. L'équipe qui gagne la donne marque 160+ 2 fois le contrat, l'équipe qui perd marque 0.

Pour compter à tout atout, il y a 248 points en tout, pas de der. Pour faire 80-TA, l'attaque doit faire 120 points à TA. Puis le comptage se fait de 15 en 15: pour 90-TA, l'attaque doit faire 135, pour 100-TA, l'attaque doit faire 150, et ainsi de suite, les correspondances sont précisés ci-après:
Pour 80-90-100-110-120-130-140-150-160 normalement, à TA on a : 120-135-150-165-180-195-210-225-240. Le capot rest compté à 250.
A TA, les points faits encaissés par l'attaque ou la défense sont arrondis à l'inférieur pour l'attaque, et au supérieur pour la défense : par exemple si l'attaque fait son contrat à 120 TA et fait entre 195 et 209 points (avant conversion), alors elle marque les 120 de son contrat + les 130 points équivalents encaissés. La défense marque les 30 points de défense.

