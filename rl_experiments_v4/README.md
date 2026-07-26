# Checkpoints RL - lignee v4 (CardNetBig : MLP a plusieurs couches)

Dossier separe de `rl_experiments_v3/`, cree quand `coinche/rl_agent.py` a
gagne `CardNetBig` (167 -> 128 -> 128 -> 64 -> 32, GELU, LayerNorm Pre-LN
avant chaque couche lineaire, ~1.9x plus de parametres que `CardNet`) --
memes principe que les passages v1->v2 (regle 5 du scoring) et v2->v3
(nouvelle feature d'etat `_points_so_far`) : l'architecture change de
forme (`fc1` a une shape differente), donc les checkpoints ne sont plus
compatibles avec `CardNet` et les lignees precedentes. L'etat/scoring ne
changent pas (toujours 167-dim, meme moteur de jeu) : `CardNetBig` reste
chargeable via `--architecture big` (`train.py`/`pretrain.py`/
`eval_policy.py`/`eval_matchup.py`, cf. `PPOPolicy.policy_net_cls`/
`NeuralPolicy.net_cls`), `CardNet` (`--architecture small`, defaut) reste
inchange et les checkpoints de `rl_experiments_v3/` restent charges et
reproductibles a l'identique.

Contexte/plan complet : `coinche/remarques_rl.md` point 6. Le pool
grandissant (`rl_experiments_v3/README.md`) est la seule piste
structurelle qui a tenu cette session ; toutes les pistes cote critic ont
echoue parce que `ValueNet` est un reseau totalement independant de
`CardNet` -- rien ne peut atteindre l'acteur. Etape 1 du plan (celle-ci) :
tester si la capacite seule (`CardNet` n'a qu'une couche cachee de 128)
est un facteur limitant, isolee de tout le reste (reseaux toujours
separes, entrainement simple contre heuristic, REINFORCE d'abord).

## `imit_bignet.pt` : accuracy d'imitation nettement meilleure

Meme protocole que `rl_experiments_v3/imit.pt` (5000 donnes, 20 epochs,
seed 0) : **98.2% d'accuracy d'imitation, contre 92.8% pour `imit.pt`**
(`CardNet`) -- signal fort que la capacite du petit reseau etait
limitante pour imiter l'heuristique.

## Balayage de lr, REINFORCE, seed20 (100k episodes)

Fine-tuning REINFORCE depuis `imit_bignet.pt` (`--architecture big`),
sinon meme protocole que `rl_experiments_v3/exp_reinforce_lr3e-5`
(`entropy-beta=0.002, entropy-decay=none, opponent=heuristic`) :

| lr | eval_avg(45000) | eval_avg(10000, echantillon different) |
|---|---|---|
| `3e-5` | +11.51 | +15.22 |
| `1e-4` | +14.06 | +16.38 |
| `3e-4` | -27.9 (eval_avg(500) en fin d'entrainement) | **-28.91 (policy detruite)** |

Std par partie (sur l'echantillon a 10000) ~228.5, erreur-type ±2.29 a
n=10000 -- l'ecart entre les deux mesures du MEME checkpoint lr=3e-5
(+11.51 puis +15.22 selon l'echantillon de donnes) illustre le bruit de
mesure residuel meme a n=45000 (erreur-type ~1.08 sur cet echantillon-la).

**Constats (1 seed, prudence) :**
- `lr=3e-4` detruit clairement la policy (des dizaines d'erreurs-types
  sous les deux autres) -- meme pathologie que `8e-4` pour PPO
  (`coinche/remarques_rl.md` point 5).
- `lr=3e-5` vs `lr=1e-4` : l'ecart observe (1.16 a 2.55 points selon
  l'echantillon) est dans le bruit (erreur-type ~2.29) -- pas de
  difference claire entre les deux a ce stade.
- Sur le dernier echantillon (n=10000), les deux configs non-divergees
  (+15.22, +16.38) se retrouvent **autour du plateau etabli pour
  `CardNet`** (`rl_experiments_v3` : PPO `epochs=8` 15.13, REINFORCE
  `lr=3e-5` 15.15, sur plusieurs seeds) -- ni clairement meilleur ni
  clairement pire, contrairement a ce que suggerait la toute premiere
  mesure (+11.51 seul, a l'epoque un seul point de donnee).

## `lr=1e-4`, n=3 : stable, mais legerement sous le plateau `CardNet`

3 seeds (20/21/22, meme protocole, `lr=1e-4`) evalues a n=45000 :

| seed | eval_avg(45000) |
|---|---|
| 20 | +14.06 |
| 21 | +14.00 |
| 22 | +12.87 |

Moyenne **13.64**, ecart-type **0.67** -- tres stable entre seeds (dans
le meme ordre de grandeur que les configs stables de `rl_experiments_v3`,
std 0.39-0.40). Mais ~1.5 point **en dessous** des deux plateaux `CardNet`
deja etablis (PPO `epochs=8` : 15.13/std=0.39 ; REINFORCE `lr=3e-5` :
15.15/std=0.40). Test de Welch contre chacun des deux : t≈3.3-3.5
(df≈3.1-3.2, seuil critique ≈3.18) -- juste au-dessus du seuil
conventionnel, donc un ecart tout juste significatif a ce stade, mais pas
massif (n=3 contre n=3-4, comme toujours dans cette lignee a prendre avec
prudence avant de le considerer acquis).

**Portee explicite de cette conclusion (importante) :** ce resultat ne
concerne que **`lr=1e-4` sur REINFORCE simple**, pas `CardNetBig` en
general :
- `lr=3e-5` n'a jamais ete teste a plusieurs seeds (un seul point,
  +11.51 puis +15.22 selon l'echantillon d'eval -- lui-meme dans le
  bruit d'echantillonnage, cf. plus haut) -- il est possible que `3e-5`
  soit en fait meilleur que `1e-4` pour ce reseau plus profond, une fois
  confirme a plusieurs seeds.
- Cette etape ne teste que la capacite pure, sur REINFORCE, contre
  heuristic seul, avec le protocole d'imitation/donnees identique a
  `CardNet`. Reste a voir comment ce reseau apprend avec d'autres
  changements de donnees (plus de donnes/epochs a l'imitation, pool
  grandissant...) et surtout avec l'etape 2 du plan
  (`coinche/remarques_rl.md` point 6) : tronc partage acteur/critic +
  critic centralise + taches auxiliaires, jamais testee -- cette premiere
  etape isolee ne prejuge pas du resultat une fois le critic implique.

## `lr=3e-5`, n=2 (partiel) : variance plus large, pas encore de conclusion

Seed21 ajoute (meme protocole, `lr=3e-5`) :

| seed | eval_avg(45000) |
|---|---|
| 20 | +11.51 |
| 21 | +15.37 |

Moyenne 13.44 (n=2) -- proche de la moyenne `lr=1e-4` (13.64, n=3), mais
avec un ecart bien plus large entre les deux seeds (3.86 points, contre
1.19 sur les 3 seeds `lr=1e-4`) -- pas encore assez de seeds pour dire si
`lr=3e-5` a une vraie variance plus grande ou si c'est juste n=2. Le
seed21 seul (+15.37) tombe pile dans le plateau `CardNet` habituel
(~15.1-15.2). **Pas de conclusion sur lequel des deux lr est meilleur a
ce stade** -- un 3e seed a `lr=3e-5` serait necessaire pour comparer a
rigueur egale.

## Piste ouverte : la policy issue de l'imitation est-elle trop confiante ?

Hypothese testee (discussion 2026-07-26) : la legere sous-performance de
`CardNetBig` pourrait venir d'un point de depart d'imitation trop
confiant/peu plastique pour le fine-tuning REINFORCE (gradient de policy
bruite, une seule mise a jour par donne), plutot que d'un probleme de
capacite ou de vanishing gradient (peu plausible ici : reseau peu
profond, GELU + LayerNorm Pre-LN deja concus pour eviter ca, et les
trajectoires d'entrainement montrent un vrai mouvement, pas une
stagnation).

**Test A (verifie)** : entropie de la policy juste apres imitation (avant
tout RL), sur un echantillon de 16000 decisions (500 donnes heuristic x4) :

| | entropie moyenne | top1_prob moyen |
|---|---|---|
| `CardNet` (`imit.pt`) | 0.2153 (std=0.3186) | 91.5% |
| `CardNetBig` (`imit_bignet.pt`) | **0.0848** (std=0.1972) | **96.6%** |

**Confirme l'hypothese** : `CardNetBig` sort de l'imitation avec une
entropie ~2.5x plus faible -- distribution bien plus pointue (96.6% de la
masse sur le coup prefere, contre 91.5%). Coherent avec un point de
depart trop confiant, meme si ca ne prouve pas encore le lien de causalite
avec la performance RL finale.

**A tester ensuite** : `entropy-beta=0.006` (facteur ~2.5-3x l'actuel
0.002, calibre sur l'ecart d'entropie mesure -- prefere a un saut
arbitraire type 0.02, par prudence : on ne connait pas la sensibilite de
ce reseau a ce coefficient, et augmenter trop pourrait tout aussi bien
sur-corriger vers une policy trop aleatoire). Plan : mesurer d'abord
l'entropie post-entrainement des 5 checkpoints deja entraines a 0.002
(reutilise les checkpoints existants, aucun nouvel entrainement
necessaire) comme reference, puis lancer un run court (20000 episodes)
a 0.006 pour verifier que l'entropie realisee augmente comme attendu,
avant d'investir dans une comparaison complete a 100k episodes.

**A faire si on veut poursuivre** : confirmer/infirmer `lr=3e-5` a
plusieurs seeds avant de trancher entre les deux lr ; tester
`entropy-beta` plus eleve (piste ci-dessus) ; tester PPO avec
`CardNetBig` (jamais fait, seulement REINFORCE jusqu'ici) ; puis passer a
l'etape 2 du plan (tronc partage).
