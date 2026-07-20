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

## PPO (`train_ppo.py`) : comparaison a REINFORCE

`train_ppo.py` implemente PPO (Schulman et al. 2017) sur exactement la meme
tache que `train.py`/REINFORCE (meme etat/action, meme equipe RL aux sieges
0/2, meme contre-factuel du point 3 de `remarques_rl.md`), pour comparer les
deux algorithmes a protocole egal. Round 1 : `--load imit.pt --opponent
heuristic` seul (pas de pool/PFSP), 50k episodes -- echelle du segment 1
REINFORCE ci-dessus (+14.4 vs `imit.pt`).

### Deux bugs trouves et corriges avant tout resultat exploitable

Le tout premier run (`exp_ppo_round1_buggy_valueloss`) degradait la policy
en dessous de zero (eval_avg negatif tout du long, entropie qui grimpe sans
converger, 0.56 a 1.03). Deux causes independantes :

1. **Echelle de `value_loss`** : MSE sur des retours bruts (~centaines de
   points) dominait `policy_loss` (sur avantage normalise, O(1)) de
   plusieurs ordres de grandeur dans la perte totale -- le gradient qui
   traversait le tronc partage ne servait quasiment plus qu'a minimiser
   l'erreur de valeur. Fix : diviser `values`/`returns` par l'ecart-type du
   batch avant le MSE.
2. **Masque des coups legaux non reapplique** (`exp_ppo_round1_buggy_mask`,
   apres le fix #1 -- toujours degrade, pire meme) : `choose_card`
   echantillonne sous une distribution masquee (coups illegaux a `-inf`),
   mais `update_batch` recalculait `new_log_probs` sous les logits **non
   masques** -- le ratio PPO (`exp(new_log_prob - old_log_prob)`) n'avait
   plus le sens "combien la policy a-t-elle bouge". Fix : stocker le masque
   dans la trajectoire, le reappliquer au recalcul. Verifie par un test
   direct : ratio dans [0.999999, 1.000002] juste apres collecte (avant tout
   gradient step), comme attendu si le masque est coherent des deux cotes.

Round 1 refait avec les deux fix (`exp_ppo_round1_epochs4`) : enfin sain
(entropie stable ~0.2, `value_loss` stable ~1.0 normalise), eval_avg(3000)
via `eval_policy.py` = **+7.84** contre **+15.22** pour REINFORCE
(`seg_50000`) au meme budget -- net progres sur `imit.pt` (-3.08) mais a
peine au-dessus du bruit structurel de `heuristic` contre lui-meme (+7.38).

### Ablation `--epochs` (reutilisation du batch)

Avec `epochs=4, minibatch=64`, le nombre de gradient steps par episode
(`epochs x 16 / minibatch`, 16 = timesteps enregistres par donne, 2 sieges
RL x 8 plis) tombe a 1.0 -- autant que REINFORCE, ce qui sous-exploite
l'avantage propre a PPO (reutiliser un batch plusieurs fois grace au
clipping). `epochs=8` (2 steps/episode) : eval_avg=**+9.75**
(`exp_ppo_round1_epochs8`) -- mieux, mais le gain (~2 pts) reste sous l'IC95%
empirique (~+/-7-8 pts, n=3000) : pas significatif sur un seul run.

### Piste exploree et abandonnee (pour l'instant) : pre-entrainement du critic

Diagnostic (voir aussi `remarques_rl.md`) : le critic a tronc partage
(`epochs8`) n'expliquait que R2=0.043 de la variance du retour brut, et
separait a peine attaque/defense (`is_attacker`, pourtant une feature
d'entree directe) : ~16 points captes sur un ecart reel de ~197. Cause
probable : son tronc, herite de `imit.pt`, n'avait jamais vu de signal de
valeur avant PPO, en concurrence avec `policy_loss` sur ce meme tronc
partage.

Fix architectural : `ActorCriticNet` (tronc partage) remplace par `CardNet`
(policy, = `imit.pt`, aucun remappage requis au chargement) + `ValueNet`
(tronc **independant**), plus de concurrence de gradient. `pretrain_value.py`
(nouveau, calque sur `pretrain.py`) pre-entraine `ValueNet` par regression
MSE supervisee.

- **v1 (bug de cible)** : regresse sur le retour **brut**
  (`team_points[0]-team_points[1]`, donnes `HeuristicPlayer` x4) -- bon R2
  hors ligne (0.33, 5000 donnes/20 epochs), mais `value_loss` en PPO porte
  en realite sur `reward - counterfactual_reward(...)` (deja applique dans
  la boucle principale avant que le critic n'intervienne, cf. point 3).
  Verifie empiriquement : l'ecart attaque/defense tombe de ~190 points sur
  le retour brut a ~17 points seulement sur le retour contre-factuel -- ce
  dernier absorbe deja l'essentiel du signal facile. Le critic v1 demarrait
  donc avec des predictions confiantes (std(value)=127) mais
  **systematiquement biaisees** pour la vraie cible PPO. Resultat round PPO
  (`exp_ppo_round1_valuepretrain`) : eval_avg=**+4.92**, pire que
  epochs4/epochs8.
- **v2/v3 (cible corrigee)** : regresse sur le retour contre-factuel,
  collecte avec la policy aux sieges 0/2 (comme `run_episode`) plutot que
  `HeuristicPlayer` x4. Signal residuel bien plus faible (R2~0.05 a 5000
  donnes/20 epochs) -- confirme reel mais data-starved par un test
  train/val a 20000 donnes/60 epochs (R2(val) monte proprement de 0.008 a
  0.063, pas du bruit). `value_imit.pt` final (sur disque) : 20000 donnes,
  60 epochs, moyennes par groupe non biaisees mais mesure de R2 bruyante sur
  petit echantillon frais (~-0.05 sur 800 donnes de validation
  independantes -- attendu vu la faiblesse du signal). Resultat round PPO
  (`exp_ppo_round1_valuepretrain_v3`) : eval_avg=**-0.34** -- pire que le
  critic a init aleatoire (`epochs8`, +9.75).

**Conclusion (piste mise en pause) :** meme correctement cible et non
biaise, un critic avec un signal aussi faible (R2~0.06) peut faire plus de
mal que de bien -- ses erreurs propres (std(value) mesure entre 32 et 57)
ajoutent potentiellement plus de bruit a l'avantage qu'un critic quasi nul
(init aleatoire, proche de "pas de baseline"). Mais un seul run par
configuration (50k episodes, seed unique) ne permet pas de trancher entre
"signal reel mais nuisible" et "bruit de run" (les runs de cette session
vont de -0.34 a +15.22 pour des configs proches, du meme ordre que l'IC95%
empirique) -- necessiterait plusieurs seeds pour conclure fermement.
Architecture (`ValueNet` independant, `pretrain_value.py --load-value`) et
checkpoints conserves pour reprendre cette piste plus tard si besoin.

### Resultats consolides (`eval_policy.py`, 3000 parties, seed=42)

| Checkpoint | eval_avg(3000) | win_rate |
|---|---|---|
| seg_50000 (REINFORCE, reference) | +15.22 | 53.2% |
| **exp_ppo_round1_epochs8** (meilleur PPO a ce jour) | **+9.75** | 52.2% |
| exp_ppo_round1_epochs4 | +7.84 | 51.8% |
| heuristic (reference bruit structurel) | +7.38 | 51.8% |
| exp_ppo_round1_valuepretrain (critic pre-entraine, cible buggee) | +4.92 | 51.2% |
| exp_ppo_round1_valuepretrain_v3 (critic pre-entraine, cible corrigee) | -0.34 | 50.3% |
| imit.pt (point de depart) | -3.08 | 49.6% |

A 50k episodes et `--opponent heuristic` seul, PPO n'a pas encore rattrape
REINFORCE au meme budget. Prochaine etape : tuning d'autres hyperparametres
(`lr`, `clip-eps`, `batch-episodes`) en repartant de `epochs8` (critic a
init aleatoire, la meilleure config a ce jour), sans pre-entrainement du
critic pour l'instant.
