# Checkpoints RL - lignee v2 (post regle 5)

Dossier separe de `rl_experiments/`, cree quand le scoring de
`GameEngine.play()` (coinche/game.py) a ete corrige pour implementer la
regle 5 complete de `regles_coinche.md` : chute (attaque 0 / defense
160+contrat, au lieu de simplement rendre les points de plis bruts sans
tenir compte de la reussite du contrat), arrondi a la dizaine, bonus capot
500, echelle TA (x1.5).

Tous les checkpoints de `rl_experiments/` (`imit.pt` a `exp_pool_350k`) ont
ete entraines contre l'**ancien** signal, qui ne penalisait pas
specifiquement une chute au-dela du deficit naturel de points de cartes --
ils restent valides comme trace de cette premiere lignee (voir son
README), mais ne sont plus la reference pour de nouveaux entrainements.

`imit.pt` (imitation supervisee de `HeuristicPlayer`, independante du
reward donc toujours valide) reste le point de depart commun. Toute
nouvelle experience RL (self-play, pool, PFSP...) repart d'ici, sur le
signal corrige.

## `exp_growing_pool/` : pool grandissant + PFSP (`run_growing_pool.py`)

Orchestration en segments (voir `run_growing_pool.py` a la racine) : chaque
segment ajoute son propre checkpoint final au pool d'adversaires du segment
suivant, avec `train.py --pfsp` pour biaiser le tirage vers l'adversaire du
pool actuellement le plus coriace (win-rate le plus bas du trainee contre
lui) plutot qu'un tirage uniforme.

Reglages retenus (discussion remarques_rl.md) :
- 50k episodes par segment
- `--pfsp-refresh-every 1000`, `--pfsp-temperature 0.1`, `--pfsp-ema-beta 0.98`
- `lr=1e-4`, `entropy-beta=0.002`, `entropy-decay=none` (facteurs deja valides
  dans la lignee v1 -- voir l'ablation lr/entropie de `rl_experiments/README.md`)

Le detail (nombre de segments effectues, resultats) sera documente ici au
fur et a mesure.

### Resultats : run complet 500k (10 segments de 50k, 2026-07-20)

Evaluation avec `eval_policy.py` (donnes fixes, comparaison appariee, 3000
parties, seed=42) de `heuristic`, `imit.pt` et chaque checkpoint de segment
contre `HeuristicPlayer` :

| Checkpoint | avg vs heuristic | win_rate | delta vs imit.pt |
|---|---|---|---|
| heuristic (reference) | +2.4 +/- 8.3 | 50.7% | -- |
| imit.pt | +0.2 +/- 8.2 | 50.3% | -- |
| seg_50000 | +14.6 | 53.1% | +14.4 (signif.) |
| seg_100000 | +14.2 | 53.3% | +14.0 (signif.) |
| seg_150000 | +16.4 | 53.7% | +16.2 (signif.) |
| seg_200000 | +21.1 | 54.7% | +21.0 (signif.) |
| seg_250000 | +24.4 | 55.1% | +24.2 (signif.) |
| **seg_300000** | **+26.0** | **55.7%** | **+25.8 (pic)** |
| seg_350000 | +18.1 | 54.0% | +17.9 (chute signif. vs seg_300000 : -7.95) |
| seg_400000 | +22.6 | 54.9% | +22.4 |
| seg_450000 | +23.7 | 55.1% | +23.5 |
| seg_500000 (final) | +23.5 | 55.0% | +23.3 |

IC95% ~ +/-7 a 8 points sur ces deltas (std empirique ~228 points/donne,
n=3000 -- bien plus eleve que les ~124 de la lignee v1 pre-regle-5 : la
regle 5 introduit des ecarts beaucoup plus violents, capot 500, coinche
double, chute asymetrique).

**Constats :**
- `imit.pt` seul ne bat pas `heuristic` (attendu, c'est un clone de
  l'heuristique).
- Le **segment 1** (50k episodes, uniquement vs heuristic, pool a 1 membre)
  capture a lui seul le plus gros du gain (+14.4, hautement significatif) --
  le saut le plus net de toute l'experience.
- A partir du segment 2 (pool grandissant + PFSP actif), les progres
  segment-a-segment sont presque tous **non significatifs** (bruit de +/-6
  a 8 points) : pas de progression fiable et reguliere une fois le pool en
  jeu.
- Pic a `seg_300000` (episode 300k), puis chute significative a
  `seg_350000` (-7.95, seul delta segment-a-segment vraiment significatif en
  negatif) -- instabilite reelle, pas juste du bruit d'evaluation.
- Le point final `seg_500000` est significativement meilleur que
  `seg_50000` (+8.8) et que `imit.pt` (+23.3) -- il y a donc un progres net
  sur l'ensemble du run.
- Mais `seg_500000` n'est **pas** significativement meilleur que
  `seg_200000` (+2.3, non signif.) et est meme legerement (non
  significativement) en dessous du pic `seg_300000` (-2.6).

**Interpretation :** la quasi-totalite du gain net exploitable s'est jouee
dans les ~200-300 premiers k episodes ; les 200-300k derniers (segments 5 a
10, pool passant de 5 a 11 membres) n'ont pas produit d'amelioration
statistiquement detectable, avec meme un accroc net vers 350k. C'est mieux
que les plateaux catastrophiques de `exp_selfplay_frozen150k`/
`exp_pool_350k` (pas d'effondrement), mais l'hypothese "pool grandissant +
PFSP evite le plafonnement" n'est pas clairement confirmee par ces chiffres
au-dela de ~300k.
