# Checkpoints RL - lignee v3 (etat 167 dims : points de plis engranges)

Dossier separe de `rl_experiments_v2/`, cree quand `coinche/rl_agent.py`
`encode_state` a gagne 2 scalaires (`STATE_DIM` 165 -> 167) : points de
cartes deja engranges par mon camp / l'adversaire dans les plis complets de
la donne (`_points_so_far`), normalises par un total de reference (162).
Absents jusqu'ici de l'etat (remarques_rl.md), alors que c'est un signal
direct pour le critic PPO (approche la grandeur meme qu'il regresse) --
n'inclut ni la belote ni le 10 de der (cf. commentaire de `_points_so_far`),
seulement les points de cartes bruts des plis deja remportes.

Ce changement de dimension d'entree **casse la compatibilite** de tous les
checkpoints de `rl_experiments/` et `rl_experiments_v2/` (`fc1` change de
forme, `size mismatch` au chargement) -- d'ou la nouvelle lignee, meme
principe que `rl_experiments_v2/` quand la regle 5 du scoring avait change.

`imit.pt` regenere ici (`pretrain.py --games 5000 --epochs 20 --seed 0`,
memes reglages que la lignee precedente) : 92.8% d'accuracy d'imitation de
`HeuristicPlayer` (contre un chiffre non enregistre pour l'ancien `imit.pt`,
pas directement comparable).

## Premier test : la config PPO gagnante de la lignee v2 (`lr=3e-5, epochs=8`)

Meme protocole que `rl_experiments_v2` (`--opponent heuristic` seul, 50k
episodes, memes hyperparametres) depuis ce nouvel `imit.pt`, pour tester si
la nouvelle feature aide.

| | eval_avg(3000) | win_rate | delta vs (son) imit.pt |
|---|---|---|---|
| heuristic (reference) | +7.38 | 51.8% | -- |
| imit.pt (v3, avec la feature) | +1.28 | 50.4% | -- |
| **exp_ppo_lr3e-5** (v3, avec la feature) | **+20.41** | **54.6%** | **+19.13** |
| *pour reference : exp_ppo_round1_lr3e-5 (v2, sans la feature)* | *+17.60* | *53.8%* | *+20.68* |

**Lecture prudente** : en valeur brute (le chiffre qu'on a suivi tout du
long), c'est mieux (+20.41 vs +17.60). Mais `imit.pt` lui-meme est aussi
meilleur avec la feature (+1.28 vs -3.08 avant) -- probablement parce que la
meme info aide aussi l'imitation supervisee de l'heuristique, pas seulement
le fine-tuning PPO. En delta par rapport a son propre point de depart (la
mesure utilisee partout ailleurs dans cette session), c'est en fait
legerement **en dessous** (+19.13 vs +20.68) -- et cet ecart (~1.5 pt) est de
toute facon tres en dessous de l'IC95% empirique (~+/-7-8 pts, un seul run
chacun). Pas de preuve solide d'un gain net a ce stade : intuition bien
motivee, implementation verifiee, resultat compatible avec "aide un peu" ou
"neutre", pas encore de quoi trancher. Necessiterait plusieurs seeds (meme
remarque que pour l'ablation `--lr` de `rl_experiments_v2/README.md`) pour
conclure.

## A refaire dans cette lignee si on veut poursuivre

- REINFORCE n'a pas encore de reference dans cette lignee (`seg_50000`
  equivalent a refaire avec le nouvel etat) -- necessaire pour toute
  comparaison PPO vs REINFORCE valide ici.
- Plusieurs seeds sur `imit.pt` et sur la config PPO pour distinguer signal
  reel de bruit de run (cf. discussion generale sur la rigueur statistique).
