# Remarques sur ce que j'ai compris de rl_agent.py et comment l'améliorer

## 1) Les états
L'espace des 165 états pris en entrée du réseau de neurones qu'on veut apprendre me semble mal choisi. Il ne prend probablement pas vraiment en compte la symétrie entre chaque couleur, et essaie d'apprendre avec tellement de données qu'elles ne reflètent pas vraiment ce qui est perçu par un joueur réel. Faut-il changer l'espace d'état pour seulement prendre en compte les données naturelles (longues, annonces, ordre de jeu, impasses, etc) ? Faut-il limiter les états en entrée ou laisser le réseau gérer ça ?

## 2) Le réseau de neurones
Le réseau de neurones pourrait permettre de prendre en compte les données dont je parle en 1) si ce ne sont pas les états qui le prennent en compte. Faut-il changer sa structure ? Si oui comment ?

## 3) Le renforcement
La récompense utilisée pour le renforcement est la quantité "reward" après une partie. Etant donné le style de jeu très déterministe et réglé de la coinche, peut-être faudrait-il en fait comparer cette reward avec la reward qu'on aurait eu si on avait joué en Heuristic plutot que RL, et utiliser comme vraie récompense cette différence de reward. Ou alors ce concept est déjà plus ou moins pris en compte dans la baseline ? Ou alors ce serait une bonne façon d'initier la baseline, ou encore c'est à faire en plus de la baseline ?


## 4) La politique initiale
Plutôt que partir d'un choix random de cartes initialement, ce qui est le cas si j'ai bien compris, ne serait-ce pas mieux de partir d'un choix proche de l'heuristique + une randomisation autour de ça, afin d'espérer partir d'un choix plus raisonnable ?

## 5) Comparaison PPO / REINFORCE : des hyperparamètres repris tels quels sans les revalider
Pour `train_ppo.py`, j'ai repris `lr=1e-4` de REINFORCE (l'ablation lr/entropie de `rl_experiments/README.md`) comme point de départ, pour comparer à protocole égal. Mais un balayage de `--lr` côté PPO montre que `3e-5` est nettement meilleur que `1e-4` (voir `rl_experiments_v2/README.md`) — et surtout que ce n'est pas monotone (`8e-4` détruit la policy, `1e-5` sous-apprend). Or ce `lr=1e-4` pour REINFORCE vient lui-même d'une ablation faite *avant* la correction de la règle 5 du scoring, qui a fait passer l'écart-type empirique du reward de ~124 à ~228 points/donne (signal bien plus bruité maintenant). Rien ne garantit que ce réglage reste optimal pour REINFORCE sous le nouveau signal. Avant de conclure quoi que ce soit de définitif sur PPO vs REINFORCE, ne faudrait-il pas refaire la même ablation lr/entropie sur REINFORCE (`train.py`) sous le signal post-règle-5, plutôt que de comparer PPO fraîchement tuné à un REINFORCE dont le réglage n'a jamais été revérifié ?
