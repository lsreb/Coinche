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
