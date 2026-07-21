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

## Continuation a 100k episodes

`exp_ppo_lr3e-5/ppo_50000.pt` prolonge de 50k episodes supplementaires
(`--load`+`--load-value` sur ce meme checkpoint pour reprendre policy ET
critic, `--episode-offset 50000`, `--seed 21` -- memes hyperparametres
sinon), pour voir si la performance a 50k etait un plateau ou un instantane
precoce (cf. `exp_growing_pool` qui avait continue a progresser jusqu'a
~300k). Comparaison ci-dessous a 5000 parties (au lieu de 3000, pour un peu
plus de precision) :

| | eval_avg(5000) | win_rate |
|---|---|---|
| heuristic (reference) | +8.07 | 51.7% |
| imit.pt | +1.91 | 50.4% |
| ppo_50000 (remesure a n=5000) | +17.90 | 53.9% |
| **ppo_100000** | **+19.98** | **54.2%** |

(Les chiffres `heuristic`/`imit.pt`/`ppo_50000` different legerement de la
section precedente -- mesures a n=3000 vs n=5000, jeux de donnes differents,
pas un changement reel.)

**Lecture** : delta de +2.08 entre 100k et 50k. Avec un ecart-type empirique
~228-230 pts/donne, l'IC95% a n=5000 est de l'ordre de +/-6 points -- ce
delta reste dedans, pas distinguable du bruit. A prendre aussi avec la meme
reserve que partout ailleurs cette session : le segment 50k->100k a un seed
d'entrainement different (21 vs 20) du premier segment, donc "plus
d'episodes" et "seed different" sont confondus, un seul run ne permet pas de
les separer. **Conclusion : stable, pas de progres net detecte entre 50k et
100k** -- cohérent avec un plateau atteint plutot qu'une tendance claire a
continuer de s'ameliorer, mais pas prouve statistiquement pour l'instant.

L'entrainement PPO ne sauvegarde pas l'etat de l'optimiseur Adam (moments
m/v) entre deux segments -- chaque reprise (`--load`/`--load-value` dans un
nouveau process) redemarre Adam "a froid" sur des poids deja entraines.
Impact estime faible (quelques centaines de steps perturbes sur les ~100k
de ce segment) et deja tolere dans la lignee REINFORCE par segments ; voir
le docstring de `PPOPolicy.save()` (`train_ppo.py`) pour le detail.

## Reference REINFORCE sur cette lignee (etat 167-dim)

Deux runs REINFORCE (`train.py`, `CardNet`/`NeuralPolicy`) depuis ce meme
`imit.pt`, 100k episodes chacun, `--opponent heuristic` (pas de pool/PFSP),
`entropy-beta=0.002` constant (visible dans les logs) -- memes deux valeurs
de `lr` que l'ablation PPO (`rl_experiments_v2/README.md`), pour tester
directement la mise en garde methodologique de `remarques_rl.md` point 5 :
le `lr=1e-4` de REINFORCE n'avait jamais ete revalide apres la correction de
la regle 5 du scoring.

## Comparaison complete (`eval_policy.py`, 5000 parties, seed=42)

| Checkpoint | eval_avg(5000) | win_rate |
|---|---|---|
| **REINFORCE lr=3e-5, 100k** | **+25.14** | **55.3%** |
| REINFORCE lr=1e-4, 100k | +21.07 | 54.3% |
| PPO lr=3e-5, 100k | +19.98 | 54.2% |
| PPO lr=3e-5, 50k | +17.90 | 53.9% |
| heuristic | +8.07 | 51.7% |
| imit.pt | +1.91 | 50.4% |

**Constat central : REINFORCE reprend la tete une fois traite equitablement.**
La conclusion "PPO bat REINFORCE" de `rl_experiments_v2` (+17.60 PPO vs
+15.22 `seg_50000`) reposait sur un REINFORCE dont le lr (1e-4) n'avait
jamais ete revalide apres la regle 5 -- exactement la mise en garde deja
documentee a l'epoque. Ici, a traitement egal (meme etat, meme lr teste des
deux cotes, memes 100k episodes), REINFORCE `lr=3e-5` (+25.14) depasse
nettement PPO `lr=3e-5` 100k (+19.98) ; meme REINFORCE `lr=1e-4` (+21.07),
deja superieur a PPO, confirme que baisser le lr aide aussi REINFORCE (delta
+4.07 par rapport a son propre `lr=1e-4`, meme sens que l'ablation PPO --
mais cet ecart reste lui aussi sous l'IC95% empirique a n=5000, ~+/-6 pts :
suggestif, pas prouve, comme tout le reste de cette session sur un seul
seed).

**A retenir** : le facteur qui domine tous les ecarts observes cette session
n'est ni l'algorithme (PPO vs REINFORCE) ni la nouvelle feature d'etat --
c'est le `lr`, qui n'avait ete correctement calibre pour aucun des deux
avant cette derniere serie de tests.

## A refaire dans cette lignee si on veut poursuivre

- Plusieurs seeds (`imit.pt`, PPO, REINFORCE) pour distinguer signal reel de
  bruit de run -- aucune des comparaisons de cette lignee n'a ete repetee.
- Isoler proprement l'effet de la nouvelle feature d'etat : comparer PPO et
  REINFORCE avec et sans la feature, a lr et nombre d'episodes egaux, sans
  confondre avec le changement de lr ou de dimension d'etat comme c'est le
  cas dans les chiffres ci-dessus.
