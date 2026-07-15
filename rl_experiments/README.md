# Checkpoints RL - point 3/4 de remarques_rl.md

`imit.pt` : policy `CardNet` pre-entrainee par imitation supervisee de
`HeuristicPlayer` (`pretrain.py --games 5000 --epochs 20 --seed 0`), point de
depart commun aux trois experiences ci-dessous.

Trois runs de fine-tuning REINFORCE (`train.py --load imit.pt --episodes 100000
--eval-every 5000 --eval-games 500 --entropy-beta 0.02 --seed 5`, contre-factuel
heuristique actif par defaut, cf. point 3), ne differant que par le schema de
decroissance du bonus d'entropie (`--entropy-decay`) :

- `exp_none/` : bonus d'entropie constant (`entropy_beta=0.02` tout du long).
- `exp_invsqrt/` : decroissance en `beta/sqrt(episode)`.
- `exp_inv/` : decroissance en `beta/episode`.

Chaque dossier contient un checkpoint tous les 5000 episodes
(`ckpt_ep5000.pt` ... `ckpt_ep100000.pt`), `final.pt` (identique a
`ckpt_ep100000.pt`), et `log.txt` (sortie complete de l'entrainement : reward
brut, reward contre-factuel, eval gloutonne, taux de victoire, beta effectif).

Constat (voir la conversation) : aucune des trois n'a depasse la performance
de `imit.pt` seul sur une evaluation robuste (1500 parties) ; les trois
oscillent sans converger clairement sur 100k episodes. Le detail chiffre est
dans `log.txt` de chaque dossier.
