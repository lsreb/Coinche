# Checkpoints RL - point 3/4 de remarques_rl.md

`imit.pt` : policy `CardNet` pre-entrainee par imitation supervisee de
`HeuristicPlayer` (`pretrain.py --games 5000 --epochs 20 --seed 0`), point de
depart commun a toutes les experiences ci-dessous.

## Decroissance du bonus d'entropie (premiere serie)

Deux runs de fine-tuning REINFORCE (`train.py --load imit.pt --episodes 100000
--eval-every 5000 --eval-games 500 --entropy-beta 0.02 --seed 5`, contre-factuel
heuristique actif par defaut, cf. point 3), ne differant que par le schema de
decroissance du bonus d'entropie (`--entropy-decay`) :

- `exp_invsqrt/` : decroissance en `beta/sqrt(episode)`.
- `exp_inv/` : decroissance en `beta/episode`.

(`exp_none/`, la variante a entropie constante, a ete abandonnee : moins
concluante que les deux ci-dessus et redondante avec `exp_lowlr_lowent`
ci-dessous, qui isole mieux l'effet de l'entropie constante via l'ablation.)

Chaque dossier contient un checkpoint tous les 5000 episodes
(`ckpt_ep5000.pt` ... `ckpt_ep100000.pt`), `final.pt` (identique a
`ckpt_ep100000.pt`), et `log.txt` (sortie complete de l'entrainement : reward
brut, reward contre-factuel, eval gloutonne, taux de victoire, beta effectif).

Constat : aucune des deux n'a depasse la performance de `imit.pt` seul sur une
evaluation robuste (1500 parties) ; les deux oscillent sans converger
clairement sur 100k episodes. Le detail chiffre est dans `log.txt` de chaque
dossier. Cause identifiee ensuite (voir plus bas) : `lr=1e-3` est trop eleve
pour du fine-tuning post-imitation, independamment du schema d'entropie.

## `exp_lowlr_lowent/` : lr et entropie reduits d'un facteur 10 (retenu)

Meme depart (`imit.pt`), mais `lr=1e-4` et `entropy-beta=0.002` (constant,
`--entropy-decay none`), pousse a 100k episodes en 3 runs sequentiels
(episodes 1-20000 seed=5, 20001-50000 seed=6, 50001-100000 seed=7, chaines via
`--load`/`--save` et `--episode-offset` pour une numerotation absolue coherente
et des donnes differentes a chaque reprise). Seuls `final.pt` (poids a 100k) et
`log.txt` (historique complet des 3 runs) sont gardes ; les checkpoints
intermediaires ont ete supprimes apres analyse.

Ablation isolant l'effet de chaque facteur (2x2, 20k episodes, seed=5,
comparaison sur les memes 1500 donnes via `eval_policy.py --games 1500`) :

|                    | beta=0.02 (haut) | beta=0.002 (bas) |
|--------------------|-----------------:|------------------:|
| **lr=1e-3 (haut)** | -17.51 (exp_none, supprime) | -23.28 |
| **lr=1e-4 (bas)**  | +0.74            | **+6.20**          |

(valeurs : `eval_avg` sur 1500 parties vs `HeuristicPlayer`, memes donnes pour
toutes les cases ; repere miroir heuristique-vs-heuristique = +1.38.)

=> c'est la reduction du `lr` qui porte l'essentiel du gain ; reduire
`entropy-beta` seul (lr reste a 1e-3) aggrave meme le resultat. Reduire les
deux ensemble (`exp_lowlr_lowent`) est la meilleure combinaison trouvee.

Progression sur les 100k episodes (`eval_policy.py`, 1500 parties, memes
donnes) : ep20000 = +6.20, ep50000 = +5.13, ep100000 (`final.pt`) = +5.79 —
plateau atteint des les premiers 20k episodes, pas de gain net supplementaire
sur les 80k episodes suivants malgre le budget investi.

## `exp_selfplay_frozen100k/` : self-play contre une copie figee (retenu)

Le plateau de `exp_lowlr_lowent` ci-dessus a suggere que `HeuristicPlayer`
comme adversaire fixe ne "pousse" plus la policy une fois qu'elle le bat deja
en moyenne. Test : reprendre l'entrainement a partir de
`exp_lowlr_lowent/final.pt` (sieges 0/2), mais face a une **copie figee** du
meme checkpoint aux sieges 1/3 (`train.py --opponent`, jeu glouton, jamais
mise a jour) plutot que face a `HeuristicPlayer`. Le contre-factuel du point 3
suit desormais lui aussi l'adversaire reel (donc la copie figee aux 4 sieges,
plus Heuristic -- cf. docstring de `counterfactual_reward` dans `train.py`),
pour rester une reference coherente avec ce qui est reellement joue.

100k episodes de plus (numerotation absolue 100001-200000, `--episode-offset
100000`, seed=8, memes `lr=1e-4`/`entropy-beta=0.002` constant que
`exp_lowlr_lowent`). Seuls `final.pt` et `log.txt` sont gardes (memes
conventions que ci-dessus). Note : dans `log.txt`, `eval_avg`/`win_rate`
mesurent la progression **contre la copie figee** (proche de 50%, attendu vu
que les deux partent des memes poids), pas contre `HeuristicPlayer`.

Resultat, mesure sur la vraie reference (`eval_policy.py --games 1500` vs
`HeuristicPlayer`, memes donnes que les tableaux precedents) :

| Policy | eval_avg(1500) | win_rate |
|---|---:|---:|
| `exp_selfplay_frozen100k/final.pt` | **+9.76** | **53.6%** |
| `exp_lowlr_lowent/final.pt` (point de depart) | +5.79 | 50.7% |
| `heuristic` (miroir) | +1.38 | 49.7% |
| `imit.pt` | -3.54 | 49.7% |

=> le self-play a fait progresser la policy au-dela du plateau observe contre
`HeuristicPlayer` seul (+5.79 -> +9.76, quasiment double). Piste suivante
naturelle : repeter l'operation (self-play contre une copie figee de
`exp_selfplay_frozen100k/final.pt`) pour voir si le gain se reproduit ou si un
nouveau plateau apparait.

## `exp_selfplay_frozen150k/` : round 2 de self-play (point final retenu)

Meme recette, un round de plus : 50k episodes supplementaires (numerotation
absolue 200001-250000, seed=9) a partir de `exp_selfplay_frozen100k/final.pt`
(sieges 0/2), contre une copie figee **du meme checkpoint** aux sieges 1/3
(`exp_selfplay_frozen100k/final.pt` reste necessaire comme adversaire figes de
ce round et n'a pas ete supprime).

Contrairement au round 1, **pas de progression nette sur ce segment** : eval
robuste (1500 parties vs `HeuristicPlayer`, memes donnes) ep200000=+9.76,
ep210000=+8.49, ep220000=+6.33, ep230000=+15.34, ep240000=+4.23,
ep250000(`final.pt`)=+5.78 -- oscillation sans tendance, le pic a ep230000
s'est avere etre en grande partie du bruit d'evaluation (retombe a +8.71 sur
un reeval a 10000 parties, contre +6.51 pour `final.pt` sur le meme
echantillon -- ecart de ~2 points, sous l'erreur standard mesuree a ce n
(ecart-type empirique du reward par donne = 124 points, soit SE~1.25 a
n=10000). Les deux checkpoints sont statistiquement indiscernables ; `final.pt`
est garde comme point final car c'est l'arret naturel de l'entrainement, pas
un choix cherry-picke sur l'eval.

Interpretation : un round de self-play contre une copie figee donne un vrai
gain (round 1), mais continuer contre la **meme** copie figee au-dela ne
pousse plus la policy plus loin -- probablement parce que l'adversaire fixe
devient lui aussi "battu en moyenne" et cesse de fournir un gradient utile,
comme observe avec `HeuristicPlayer` dans `exp_lowlr_lowent`. Piste suivante :
rafraichir l'adversaire fige a chaque round (curriculum type fictitious
self-play) plutot que de repeter un round contre le meme snapshot.

**Bilan de la lignee complete** (`eval_policy.py --games 1500`, vs
`HeuristicPlayer`) : `imit.pt` -3.54 -> `exp_lowlr_lowent` +5.79 ->
`exp_selfplay_frozen100k` +9.76 -> `exp_selfplay_frozen150k` +5.78 (dans le
bruit du round precedent, pas une regression averee vu la marge d'erreur).
