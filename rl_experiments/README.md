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
sur les 80k episodes suivants malgre le budget investi. Prochaine piste
suggeree : le signal contre-factuel reste bruite (une seule trajectoire
heuristique de reference par donne), ou les points 1/2 de `remarques_rl.md`
(espace d'etat / architecture) limitent la policy plutot que les
hyperparametres d'optimisation.
