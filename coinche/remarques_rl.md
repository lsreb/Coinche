# Remarques sur ce que j'ai compris de rl_agent.py et comment l'améliorer

## 1) Les états
L'espace des 112 états pris en entrée du réseau de neurones qu'on veut apprendre me semble mal choisi. Il ne prend probablement pas vraiment en compte la symétrie entre chaque couleur, et essaie d'apprendre avec tellement de données qu'elles ne reflètent pas vraiment ce qui est perçu par un joueur réel. Faut-il changer l'espace d'état pour seulement prendre en compte les données naturelles (longues, annonces, ordre de jeu, impasses, etc) ? Faut-il limiter les états en entrée ou laisser le réseau gérer ça ?

## 2) Le réseau de neurones
Le réseau de neurones pourrait permettre de prendre en compte les données dont je parle en 1) si ce ne sont pas les états qui le prennent en compte. Faut-il changer sa structure ? Si oui comment ?

## 3) Le renforcement
La récompense utilisée pour le renforcement est la quantité "reward" après une partie. Etant donné le style de jeu très déterministe et réglé de la coinche, peut-être faudrait-il en fait comparer cette reward avec la reward qu'on aurait eu si on avait joué en Heuristic plutot que RL, et utiliser comme vraie récompense cette différence de reward. Ou alors ce concept est déjà plus ou moins pris en compte dans la baseline ? Ou alors ce serait une bonne façon d'initier la baseline, ou encore c'est à faire en plus de la baseline ?
