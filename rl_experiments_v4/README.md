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

## Bonus d'entropie plus fort : ne marche pas -- le probleme vient de l'imitation, pas du RL

Reference (post-entrainement, `entropy-beta=0.002`, memes 5 checkpoints
100k episodes que plus haut) : entropie 0.0838-0.1173, tres proche du
depart imitation (0.0848) -- l'entrainement REINFORCE ne bouge quasiment
pas l'entropie. Pour comparaison, `CardNet` (v3) reste toute sa vie dans
0.17-0.22 (imitation comme apres RL) -- `CardNetBig` opere a peu pres a
moitie de ce niveau, a chaque etape.

Teste `entropy-beta` plus eleve pour compenser, sur des runs COURTS
(20000 episodes, meme seed20, `lr=1e-4`, pour isoler l'effet du
coefficient a volume d'entrainement egal) :

| entropy-beta | entropie moy (20000 ep) | top1_prob moy |
|---|---|---|
| `0.002` (reference) | 0.0966 | 96.10% |
| `0.006` (~3x) | 0.0986 | 95.98% |
| `0.06` (~30x) | **0.1018** | 95.91% |

**Le bonus d'entropie n'a quasiment aucune prise ici** : meme multiplie
par 30x, l'entropie bouge a peine (+0.005 par rapport a la reference).
Hypothese : pres d'une distribution quasi-deterministe (comme celle de
`CardNetBig` des la sortie d'imitation), le gradient de l'entropie par
rapport aux logits devient tres plat -- aucun coefficient raisonnable ne
peut la faire bouger significativement une fois la policy aussi confiante.
Le probleme vient donc de l'**imitation elle-meme**, pas d'un correctif a
appliquer apres coup pendant le RL.

## Arreter l'imitation a une entropie cible (0.20, niveau de `CardNet`)

Plutot que de corriger apres coup, arrete l'entrainement d'imitation des
que l'entropie de la policy (mesuree sur le meme echantillon de 500
donnes de reference) atteint la cible de `CardNet` (~0.20), au lieu
d'aller jusqu'a convergence complete (20 epochs, ou l'entropie tombe a
0.0848).

| epoch | accuracy | entropie (probe) |
|---|---|---|
| 1 | 82.1% | 0.3827 |
| 2 | 89.0% | 0.2933 |
| 3 | 90.9% | 0.2421 |
| 4 | 92.1% | 0.2149 |
| **5** | **92.9%** | **0.1922** (cible atteinte) |

Arrete a l'epoch 5 -> `rl_experiments_v4/imit_bignet_ent02.pt`. Fait
notable : l'accuracy a ce point (92.9%) est quasiment identique a celle
de `CardNet` totalement converge (92.8%) -- les deux reseaux atteignent
le meme niveau d'imitation a une entropie comparable, `CardNetBig` juste
beaucoup plus vite (5 epochs contre 20) grace a sa capacite
supplementaire. Meme architecture que `imit_bignet.pt` (aucune rupture de
compatibilite, reste dans cette lignee v4, pas une nouvelle lignee --
seul le point d'arret de l'entrainement change, pas la forme des poids).

## REINFORCE depuis `imit_bignet_ent02.pt` (lr=3e-5) : gain massif, CONFIRME a n=3

Meme protocole exactement que les runs precedents (`lr=3e-5,
entropy-beta=0.002, entropy-decay=none, opponent=heuristic, 100k
episodes`), mais depuis `imit_bignet_ent02.pt` (entropie 0.192, epoch 5)
au lieu de `imit_bignet.pt` (entropie 0.0848, epoch 20).

| seed | eval_avg(45000) |
|---|---|
| 20 | +25.04 |
| 21 | +23.21 |
| 22 | +25.00 |

Moyenne **24.42**, ecart-type **1.05** (n=3) -- tres serre, un vrai
signal, pas du bruit. Pour comparaison : plateau `CardNet` etabli (PPO
`epochs=8` : 15.13/std=0.39 ; REINFORCE `lr=3e-5` : 15.15/std=0.40, tous
sur plusieurs seeds) ; `CardNetBig` depuis l'ancien `imit_bignet.pt`
(imitation a convergence complete) a `lr=3e-5` (n=2) : 11.51, 15.37
(moyenne 13.44, borderline pire que `CardNet`).

**Delta de +9.27 points par rapport au plateau REINFORCE `lr=3e-5` de
`CardNet`** -- test de Welch t≈14.6, des dizaines d'erreurs-types
au-dessus de tout seuil de significativite. C'est de loin le resultat le
plus large et le mieux confirme de toute la session (REINFORCE ou PPO,
v3 ou v4).

**Conclusion** : l'hypothese complete tenait. La capacite supplementaire
de `CardNetBig` (128/128/64, GELU, LayerNorm Pre-LN) aide bel et bien --
mais seulement si on evite le point de depart trop confiant issu d'une
imitation poussee a convergence complete (entropie 0.0848, ou le bonus
d'entropie pendant le RL n'a aucune prise, cf. sections precedentes).
En arretant l'imitation a une entropie cible (~0.20, niveau de
`CardNet`) au lieu d'aller jusqu'a convergence, le fine-tuning REINFORCE
part d'un point bien plus exploitable et debloque un gain massif et
desormais bien confirme (n=3, std=1.05).

## A faire si on veut poursuivre

- Refaire `lr=1e-4` depuis `imit_bignet_ent02.pt` (au moins 1 seed) pour
  voir si le lr compte encore une fois le bon point de depart utilise, ou
  si `lr=3e-5` domine desormais clairement.
- Tester PPO avec `CardNetBig` + `imit_bignet_ent02.pt` (jamais fait,
  seulement REINFORCE jusqu'ici) -- etant donne PPO≈REINFORCE partout
  ailleurs dans cette lignee, s'attendre a un gain similaire, mais a
  verifier.
- Essayer d'autres cibles d'entropie autour de 0.20 (ex. 0.15, 0.25) pour
  voir si le point exact compte, ou si toute la zone "pas totalement
  convergee" fonctionne pareil.
- Puis passer a l'etape 2 du plan (tronc partage acteur/critic + critic
  centralise, `coinche/remarques_rl.md` point 6) -- avec ce nouveau point
  de depart d'imitation desormais comme reference a battre.

## `lr=1e-4` depuis `imit_bignet_ent02.pt` (seed20) : bien plus faible que `lr=3e-5`

Meme protocole, seul le lr change : **eval_avg(45000) = +11.81** -- tres
en dessous de la moyenne confirmee a `lr=3e-5` (24.42, std=1.05 sur 3
seeds). Ecart (~12.6 points) largement au-dela du bruit inter-seeds
observe a `lr=3e-5`, donc probablement pas un simple seed malchanceux :
`lr=3e-5` semble specifiquement necessaire pour exploiter ce nouveau
point de depart, contrairement au petit `CardNet` ou `lr=3e-5` et
`lr=1e-4` etaient statistiquement indiscernables. Un seul seed pour
l'instant a `lr=1e-4` -- pas encore de seeds supplementaires prevus, ce
point est secondaire par rapport a la confirmation n=3 de `lr=3e-5`.

## Etape 2 du plan : tronc partage acteur/critic + critic centralise

Implemente (pas encore teste en conditions reelles) : `SharedTrunkActorCritic`
(`coinche/rl_agent.py`) et `SharedTrunkPPOPolicy` (`train_ppo.py`,
`--architecture shared`).

Architecture (discussion du 2026-07-27) :
- **Tronc partage** (acteur + critic) : les 2 premieres couches de
  `CardNetBig` (`167->128->128`, GELU + LayerNorm Pre-LN) -- memes noms
  de parametres que `CardNetBig` pour pouvoir reprendre un
  `imit_bignet_ent02.pt` deja entraine comme point de depart
  (`SharedTrunkActorCritic.load_actor_from_cardnetbig`, verifie bit a bit
  identique a `CardNetBig` sur le meme checkpoint).
- **Tete acteur** : continue seule, `128->64->32` -- identique en
  forme/noms a la 2e moitie de `CardNetBig`, comportement de la policy
  inchange a poids egaux.
- **Tete critic** : recoit en plus l'info centralisee (mains des 3
  autres, 96-dim, deja dans `encode_full_state`) via une seule couche
  separee (`96->32`, choix delibere : role plus structurel que
  strategique, discussion du 2026-07-27), concatenee a la sortie du
  tronc partage (`128+32=160`) avant 1 couche cachee privee (`160->64`)
  puis la sortie scalaire.
- Pas de taches auxiliaires dans cette premiere version (decision du
  2026-07-27 : isoler l'effet du tronc partage seul avant d'ajouter cette
  complexite).

Le point cle par rapport aux 4 tentatives precedentes sur le critic
(toutes avec un `ValueNet` totalement independant de `CardNet`) : ici le
gradient de `value_loss` traverse aussi le tronc partage, donc l'info
centralisee peut enfin influencer la representation que l'acteur utilise
-- mecanisme qui manquait par construction a toutes les tentatives
anterieures.

Smoke-teste bout en bout (chargement depuis `imit_bignet_ent02.pt`, 200
episodes d'entrainement, sauvegarde/rechargement, `eval_policy.py
--architecture shared`) -- aucune erreur.

### Premiers runs reels : interference value/policy a travers le tronc, puis correction

Premier run complet (100k episodes, `lr=3e-5`, `epochs=8, value-coef=0.5`
-- memes hyperparametres PPO que la reference `CardNet`) depuis
`imit_bignet_ent02.pt` : **eval_avg(45000) = +12.58**, nettement sous la
reference REINFORCE (24.42, n=3) et meme legerement sous le plateau
`CardNet` simple (~15.1-15.2). Trajectoire d'entrainement clairement
plus bruitee que d'habitude (`clip_frac` 5-6%, contre <2% partout
ailleurs dans cette session) -- signe d'interference value/policy a
travers le tronc partage, exactement le probleme historique qui avait
fait abandonner l'architecture PPO d'origine (`ActorCriticNet`, avant
que `policy_net`/`value_net` ne deviennent independants).

Balayage exploratoire (seed20 pour tous, `lr=3e-5` fixe) pour corriger
ca :

| epochs | value_coef | eval_avg(45000) | clip_frac (fin de run) |
|---|---|---|---|
| 8 | 0.5 | +12.58 | 5.2% |
| 4 | 0.5 | +20.53 | 1.9% |
| 4 | 0.3 | +23.81 | 1.6% |
| 4 | 0.1 | +24.42 | 1.0% |
| **4** | **0.2** | **+27.80** | 2.2% |

Les deux facteurs comptent : passer de `epochs=8` a `epochs=4` (a
`value_coef` inchange) fait deja la moitie du chemin (12.58 -> 20.53,
`clip_frac` revenu a une plage normale) -- avec un tronc partage, chaque
passe met a jour la representation depuis les deux pertes, donc 8 passes
semble sur-perturber le tronc. Baisser `value_coef` en plus ajoute un
gain supplementaire, avec un optimum apparent autour de 0.2 (0.1 et 0.3
donnent des resultats proches mais legerement inferieurs).

### Confirmation a n=4 de la meilleure config (`epochs=4, value_coef=0.2`)

| seed | eval_avg(45000) |
|---|---|
| 20 | +27.80 |
| 21 | +22.09 |
| 22 | +25.23 |
| 23 | +23.28 |

Moyenne **24.60**, ecart-type **2.50** (n=4) -- a comparer a REINFORCE
depuis le meme `imit_bignet_ent02.pt` (moyenne 24.42, ecart-type 1.05,
n=3). **Les deux moyennes sont quasiment identiques** (ecart de 0.18
point) : le tronc partage egale la meilleure config connue, sans la
depasser, avec une variance entre seeds plus large (2.50 contre 1.05).

**Conclusion** : le mecanisme fonctionne -- contrairement aux 4
tentatives precedentes sur le critic (toutes avec un `ValueNet`
independant), ici le partage de tronc ne degrade pas la performance une
fois les hyperparametres correctement recalibres pour cette architecture
(moins d'epochs, `value_coef` plus bas) -- mais aucune preuve d'un gain
net par rapport a REINFORCE simple depuis le meme point de depart, meme
a n=4. Piste suivante (cf. discussion du 2026-07-27, `rl_experiments_v5/`) :
le critic centralise a un raccourci (la branche info-centralisee) qui
dilue son propre gradient sur le tronc partage -- une tache auxiliaire
placee SEULEMENT sur le tronc (sans ce raccourci) pourrait avoir un effet
plus fort, propre a tester separement.

**A faire si on veut poursuivre** :
- Plus de seeds a `epochs=4, value_coef=0.2` pour trancher si le tronc
  partage bat reellement REINFORCE ou l'egale seulement.
- Essayer d'autres `value_coef` autour de 0.2 (ex. 0.15, 0.25) avec
  plusieurs seeds, vu qu'un seul seed par point ne permet pas de
  distinguer un vrai optimum du bruit.
- Ajouter les taches auxiliaires (atouts/AS restants des 3 autres,
  discussion initiale du plan) maintenant que le tronc partage seul
  fonctionne au moins aussi bien que REINFORCE.

## Pool grandissant depuis `imit_bignet_ent02.pt` : meilleur resultat de la session (seed40, n=1)

Apres que la piste critic/auxiliaire (tronc partage, puis tache auxiliaire
dans `rl_experiments_v5/`) n'ait rien donne de net au-dela de REINFORCE
seul, retour a l'autre resultat structurel confirme cette session : le
pool grandissant en self-play PFSP (`rl_experiments_v3/README.md`, n=3
REINFORCE + n=2 PPO, jamais teste avec `CardNetBig`/l'imitation a entropie
cible). Les deux gains n'avaient jamais ete empiles.

`run_growing_pool.py --architecture big --init-load imit_bignet_ent02.pt
--lr 3e-5 --total-episodes 300000 --segment-episodes 50000 --seed-start 40`
(6 segments, `rl_experiments_v4/exp_growing_pool_bignet/`). Corrige au
passage un bug latent dans `train.py` (`make_opponent_factory`/
`build_opponent_pool` instanciaient toujours `NeuralPolicy(net_cls=CardNet)`
pour un adversaire fige du pool, jamais un probleme tant que le pool ne
contenait que des checkpoints `CardNet` -- `load_state_dict` aurait echoue
des le segment 2 avec un pool en `CardNetBig`). Tous les 6 segments
tournes proprement (18.5 a 35 min chacun, legere hausse avec la taille du
pool ; PFSP pioche bien parmi tous les membres du pool a des win_rate
sains 0.49-0.60, aucun signe de collapse).

**eval_avg(45000) = +31.70**, win_rate=56.3% -- au-dessus de la reference
REINFORCE fixe depuis le meme point de depart (+25.04 pour ce seed
precis, moyenne 24.42/n=3) et du tronc partage (24.60, n=4). **Meilleur
resultat obtenu dans toute cette session de RL.**

**Un seul seed pour l'instant** -- l'ecart (~6-7 points au-dessus des
meilleures references) depasse la variance inter-seeds habituelle
(std 1.05 a 2.50 selon l'architecture), donc c'est prometteur, mais pas
encore confirme. **A faire avant de conclure** : au moins 2 seeds de plus
(`--seed-start 41` et `42` par exemple, memes hyperparametres) pour
confirmer que le gain tient, meme rigueur que partout ailleurs dans cette
session.

### Round-robin (`eval_matchup.py`, n=10000) : le gain se confirme, pas juste un artefact de `eval_avg`

Meme demarche que la validation du pool grandissant en v3 (cf.
`rl_experiments_v3/README.md`) : `eval_avg` contre heuristic seul ne
mesure que l'exploitation d'un adversaire fixe, pas la polyvalence.
Ajoute `--architecture` a `eval_matchup.py` (ne supportait jusqu'ici que
`PPOPolicy()` en dur/`CardNet` petit -- meme convention que
`eval_policy.py`) pour pouvoir l'utiliser sur des checkpoints
`CardNetBig`.

4 specs, 6 appariements (n=10000, seed=42) : `heuristic`, la reference
"vanille" de cette lignee (`exp_reinforce_bignet_ent02_lr3e-5_seed20/
reinforce_100000.pt`, +25.04 en `eval_avg`), le checkpoint precoce du
pool (`seg_50000`) et le checkpoint final (`seg_300000`).

Force moyenne (corrigee du biais de siege, `skill(X,Y) = ((X vs Y) -
(Y vs X))/2`, meme methode qu'en v3) :

| | force moyenne |
|---|---|
| **pool bignet, seg_300000 (final)** | **+18.97** |
| vanille (REINFORCE bignet simple) | +6.43 |
| pool bignet, seg_50000 (precoce) | +0.31 |
| heuristic | -25.71 |

**Meme pattern qu'en v3, sans exception, et avec un ecart plus marque** :
le checkpoint final du pool bat nettement la vanille en tete-a-tete
direct (+15.88 / -6.38, soit +11.13 une fois corrige du biais de siege)
et bat tres largement le checkpoint precoce (+20.08 / -9.52, soit +14.8
corrige) ; le precoce lui-meme ne se distingue pas clairement de la
vanille (+0.31 vs +6.43, dans le meme ordre de grandeur que la variance
observee en v3 entre config proches). Le gain de +31.70 en `eval_avg`
n'est donc pas (uniquement) un artefact de cette metrique bruitee -- il
se confirme en tete-a-tete direct, comme pour le pool grandissant "petit
CardNet" de v3.

**Toujours n=1** : reste a confirmer sur 2 seeds supplementaires (comme
pour v3, ou le pattern a fini par se confirmer identiquement sur 3 seeds
REINFORCE + 2 PPO) avant de traiter ce resultat comme definitivement
etabli.

## `--no-critic-baseline` sur le tronc partage (seed20) : toujours aucun effet net

Debloque `--no-critic-baseline` pour `--architecture shared` (jusqu'ici
interdit par le code) : avantage = retour centre/normalise sur le batch
(comme REINFORCE), la tete critic n'est ni appelee ni entrainee
(`value_loss` reste a 0 tout du long -- verifie). Question posee : le
tronc partage est la SEULE architecture ou le critic a un vrai canal de
gradient vers l'acteur (contrairement aux 4 tentatives anterieures avec
un `ValueNet` independant, toutes sans effet mesurable) -- est-ce que
cette fois sa baseline sert vraiment a quelque chose ?

Meme protocole exact que la reference (`lr=3e-5, epochs=4, value_coef=0.2`
-- ce dernier n'a plus d'effet ici puisque `value_loss=0`), **meme
seed20** que la reference deja connue, pour une comparaison directe :

| | eval_avg(45000) |
|---|---|
| seed20, avec critic (`value_coef=0.2`) | +27.80 |
| seed20, **sans** critic (`--no-critic-baseline`) | +24.03 |

Comparaison brute seed-a-seed : -3.77 points, l'air d'une degradation.
**Mais lue contre la vraie distribution des 4 seeds deja confirmes avec
critic (22.09 / 23.28 / 25.23 / **27.80**, moyenne 24.60, ecart-type
2.50)**, +24.03 tombe confortablement DANS cette fourchette, tres proche
de la moyenne -- et `27.80` (seed20 avec critic) est justement le PLUS
HAUT des 4 seeds connus, pas un point typique. Comparer "sans critic" a
specifiquement ce seed-la (le plus favorable au critic) plutot qu'a la
moyenne du groupe biaise la lecture vers "le critic aide" alors que rien
ne le montre clairement une fois la variance inter-seeds prise en compte.

**Lecture honnete (n=1 pour la config sans critic, prudence de rigueur)** :
aucune preuve que le critic apporte un gain net sur le tronc partage non
plus -- **7e resultat consecutif de ce type dans le projet** (apres 4
tentatives sur un `ValueNet` independant + le tronc partage lui-meme qui
egale seulement REINFORCE) : la baseline du critic ne montre son
utilite nulle part dans ce code, meme la ou elle a enfin un vrai canal
vers l'acteur. Poursuivre exigerait 2-3 seeds de plus en
`--no-critic-baseline` pour un vrai test statistique contre la
distribution n=4 existante -- pas encore fait, piste pour plus tard.

## `SharedTrunkActorCriticDeep` (tronc 3 couches, tete acteur 1 couche) : premier test, seed20

Rebalancement de `SharedTrunkActorCritic` (design exact fourni par
l'utilisateur, discussion du 2026-07-28) : tronc plus profond
(`167->128->128->64`, 3 couches -- couvre EXACTEMENT les 3 premieres
couches de `CardNetBig`) et tete acteur plus courte (`64->32`, 1 seule
couche -- la derniere de `CardNetBig`), au lieu du decoupage 2+2
actuel. Tete critic : branche centralisee `96->64` (egale a la dim de
sortie du tronc, fusion plus equilibree que l'ancien 128 vs 32), 1
couche privee `128->32`, sortie `32->1`.

Classe independante (`coinche/rl_agent.py`), jamais modifie
`SharedTrunkActorCritic` en place. `load_actor_from_cardnetbig` remappe
`imit_bignet_ent02.pt` sans aucune perte (le split tronc/tete correspond
exactement aux 4 couches de `CardNetBig`) -- verifie bit a bit identique,
donc **pas besoin de retrain l'imitation**. `SharedTrunkPPOPolicy` gagne
un parametre `net_cls` (comme `PPOPolicy`) au lieu d'une classe wrapper
dupliquee. Nouveau choix `--architecture shared_deep`.

Meme protocole exact que la reference (`lr=3e-5, epochs=4,
value_coef=0.2`, depuis `imit_bignet_ent02.pt`, seed20 -- meme seed que
les 2 autres variantes deja testees, pour comparaison directe).
Entrainement sain tout du long (`clip_frac` normal, pas de collapse).

| | eval_avg(45000) |
|---|---|
| Tronc original (2+2), seed20, avec critic | +27.80 |
| Tronc original (2+2), seed20, sans critic | +24.03 |
| **Tronc profond (3+1), seed20, avec critic** | **+23.70** |
| Tronc original, moyenne n=4 | 24.60 (std 2.50, range 22.09-27.80) |
| REINFORCE simple, moyenne n=3 | 24.42 (std 1.05) |

**Lecture honnete (n=1)** : `+23.70` tombe confortablement dans la
distribution deja etablie du tronc original (22.09-27.80), tres proche
de sa moyenne (24.60) et de celle de REINFORCE (24.42). Compare
specifiquement au seed20 original (27.80) ca semble moins bon, mais ce
seed-la est le plus haut des 4 connus (meme piege de lecture que pour
`--no-critic-baseline` ci-dessus) -- pas une comparaison juste. **Aucune
preuve que rebalancer tronc/tete change quoi que ce soit**, dans un sens
ou dans l'autre. A confirmer sur plus de seeds si on veut trancher, mais
rien d'urgent vu ce premier point neutre.

### Seed21 : confirme, tres serre autour du meme point neutre

Meme protocole exact, seed21 : **eval_avg(45000) = +22.78**.

| seed | eval_avg(45000) |
|---|---|
| 20 | +23.70 |
| 21 | +22.78 |

Moyenne **23.24**, ecart-type **0.65** (n=2) -- tres serre, les deux
seeds quasi indiscernables l'un de l'autre. Confirme la lecture du seed20
seul : `SharedTrunkActorCriticDeep` tombe pile dans la distribution du
tronc original (22.09-27.80, moyenne 24.60, n=4) et pres de REINFORCE
(24.42, n=3), sans aucun signe d'amelioration ni de degradation.
Rebalancer tronc/tete (3 couches partagees + 1 privee, au lieu de 2+2)
ne semble donc rien changer, du moins a cette echelle de test -- piste
neutre, pas prioritaire pour l'instant face au pool grandissant qui reste
la seule direction ayant clairement depasse ce plateau.

## Pool grandissant + tronc partage (original, seed-start 60) : pas de gain net, contrairement au run bignet

Apres le pool grandissant + `CardNetBig`/REINFORCE (+31.70, meilleur
resultat de la session), meme experience avec le tronc partage
(`SharedTrunkActorCritic` original, 2+2, `value_coef=0.2` -- pas la
variante `Deep`, choisie deliberement pour isoler une seule variable a
la fois par rapport a une architecture deja bien caracterisee, n=4).

`run_growing_pool_ppo.py --architecture shared --init-load
imit_bignet_ent02.pt --lr 3e-5 --epochs 4 --value-coef 0.2
--total-episodes 300000 --segment-episodes 50000 --seed-start 60` (6
segments, `rl_experiments_v4/exp_growing_pool_shared/`). Tous les
segments tournes proprement (1151-1937s, meme hausse graduelle
qu'ailleurs ; confirme au passage que le fix `SharedTrunkActorCritic.
forward()` pour charger un adversaire fige `shared` dans le pool
fonctionne en conditions reelles, pas juste au smoke-test).

| | eval_avg(45000) |
|---|---|
| Tronc partage simple (seed20, sans pool) | +27.80 |
| Pool, precoce (seg_50000) | +19.85 |
| **Pool, final (seg_300000)** | **+25.21** |
| Tronc partage simple, moyenne n=4 | 24.60 (std 2.50, range 22.09-27.80) |

**Premiere lecture (via `eval_avg` seul)** : compare au seed20 precis
(27.80), le pool final semble regresser -- mais 27.80 est encore une
fois le plus haut des 4 seeds connus (meme piege que pour les 2 lectures
precedentes de cette session). Compare a la distribution complete
(22.09-27.80, moyenne 24.60), +25.21 tombe pile dedans, essentiellement
a la moyenne : `eval_avg` seul ne montre **aucun gain net** par rapport
au tronc partage sans pool.

### Round-robin (`eval_matchup.py`, n=10000) : le gain existe bel et bien, invisible dans `eval_avg` seul

Meme demarche que pour le run bignet+REINFORCE : 4 specs (heuristic,
tronc partage simple seed20, pool precoce `seg_50000`, pool final
`seg_300000`), n=10000, seed=42. Force moyenne (corrigee du biais de
siege) :

| | force moyenne |
|---|---|
| **Pool, final (seg_300000)** | **+15.45** |
| Tronc partage simple (sans pool) | +9.00 |
| Pool, precoce (seg_50000) | -0.11 |
| heuristic | -24.34 |

**Meme pattern exceptionless qu'en v3 et que pour le run bignet** : final
> simple sans pool > precoce > heuristic. Le pool final bat la version
simple en tete-a-tete direct (+10.00 / -4.72, soit +7.36 corrige) et
ecrase le pool precoce (+16.51 / -11.11, soit +13.81 corrige) ; la
version simple bat elle-meme le precoce (+9.13 / -3.43, soit +6.28
corrige).

**Ca contredit la premiere lecture base sur `eval_avg` seul** : il y a
bel et bien un gain reel de polyvalence (+6.45 points de force corrigee
entre pool-final et simple), du meme ordre de grandeur que le gain vu
pour le pool bignet -- simplement invisible face a heuristic seul, la
lecon exacte de v3 (`eval_avg` contre un seul adversaire fixe ne mesure
que l'exploitation de cet adversaire, pas la polyvalence reelle). Le
pool grandissant apporte donc bien un gain avec le tronc partage/PPO
aussi, contrairement a ce que suggerait la premiere lecture -- **un seul
seed pour l'instant**, a confirmer sur plus de seeds avant de conclure
definitivement, meme rigueur que partout ailleurs dans cette session.

## Face-a-face direct : pool bignet+REINFORCE vs pool tronc partage+PPO (n=100000)

Les deux meilleurs resultats de la session (pool grandissant + `CardNetBig`/
REINFORCE, `eval_avg`=+31.70 ; pool grandissant + tronc partage/PPO,
`eval_avg`=+25.21 mais force de round-robin +15.45) n'avaient jamais ete
opposes directement l'un a l'autre -- chacun uniquement compare a heuristic
et a son propre groupe de reference. Ajoute le support du melange
d'architectures a `eval_matchup.py` (syntaxe `architecture:chemin` par
spec, ex. `big:seg_300000.pt shared:seg_300000.pt`, jusqu'ici un seul
`--architecture` pour tout l'appel) pour rendre ce test possible.

Round-robin a 3 specs (heuristic, `big:exp_growing_pool_bignet/seg_300000.pt`,
`shared:exp_growing_pool_shared/seg_300000.pt`), **n=100000** parties par
appariement (la precision la plus fine de toute la session, ~10x le
`--games 45000` habituel) :

| | avg |
|---|---|
| bignet-pool vs shared-pool | -0.17 |
| shared-pool vs bignet-pool | +1.06 |

Corrige du biais de siege : `skill(bignet, shared) = (-0.17 - 1.06)/2 =
-0.615`. **Statistiquement nul a cette echelle** (n=100000/sens, la plus
grande precision de toute la session) : les deux checkpoints sont a
egalite en tete-a-tete direct, malgre des `eval_avg` contre heuristic
tres differents (+31.70 vs +25.21) et des scores de force de round-robin
differents dans leurs groupes respectifs (+18.97 vs +15.45 -- mais pas
calcules contre les memes groupes de reference, donc jamais directement
comparables entre eux avant ce test).

**Conclusion** : le pool grandissant apporte un gain reel dans les deux
cas (confirme separement par round-robin pour chaque architecture), mais
une fois combine au pool, le choix de l'architecture de base (REINFORCE+
CardNetBig vs PPO+tronc partage) ne semble plus faire de difference de
niveau final -- les deux convergent vers une force de jeu similaire.

## Pool bignet+REINFORCE etendu a 450k (segments 7-9) : `eval_avg` decline, round-robin necessaire pour trancher

Meme seed (40, `--start-segment 7` sur `rl_experiments_v4/exp_growing_pool_bignet/`,
memes hyperparametres) pour voir si la progression continue au-dela de
300k. Les 3 segments supplementaires tournent proprement (2072-2108s
chacun, dans la norme).

| Segment | eval_avg(45000) |
|---|---|
| 300k | +31.70 |
| 350k | +31.28 |
| 400k | +29.53 |
| 450k | +28.70 |

**Lecture prudente** : `eval_avg` contre heuristic seul decline
graduellement apres 300k -- mais cette metrique seule est connue pour
etre trompeuse sur un checkpoint entraine en pool (lecon de v3, deja
confirmee deux fois cette session : le "gain reel" du pool grandissant
tronc-partage etait invisible dans `eval_avg`, revele seulement par
round-robin).

### Round-robin (n=10000) : le "declin" ne se confirme pas en tete-a-tete direct

4 specs (heuristic, vanille REINFORCE, pool 300k, pool 450k) :

**Tete-a-tete direct 300k vs 450k** : 300k vs 450k = -1.60, 450k vs 300k
= +8.56 -- corrige du biais de siege, **450k bat 300k de +5.08 points**.
En confrontation directe, 450k n'est donc PAS plus faible que 300k, au
contraire.

Force moyenne (contre les 3 autres) :

| | force moyenne |
|---|---|
| Pool, 300k | +15.73 |
| Pool, 450k | +11.40 |
| Vanille (REINFORCE simple) | +1.51 |
| Heuristic | -28.63 |

**Tension apparente, mais pas contradictoire** : dans la force moyenne,
300k semble devant -- tire par sa meilleure performance contre heuristic
et vanille (des adversaires plus faibles), un artefact bien connu de ce
type de classement a peu d'entites (une moyenne sur un petit groupe
d'adversaires n'est pas forcement transitive avec le tete-a-tete direct
entre les deux meilleurs). Le tete-a-tete direct reste le signal le plus
pertinent pour repondre a la question posee ("450k est-il plus faible
que 300k ?") : la reponse est non, c'est meme legerement l'inverse.

**Conclusion (corrigee)** : le "declin" de `eval_avg` ne reflete pas une
vraie regression -- encore un glissement de specialisation (moins
d'exploitation d'heuristic specifiquement). Mais contrairement a une
premiere lecture trop prudente ("essentiellement a egalite"), le
tete-a-tete montre un **vrai gain**, pas juste une absence de perte :
+5.08 corrige, coherent dans les deux sens (300k perd dans les deux
orientations du match), du meme ordre de grandeur que les autres ecarts
traites comme reels cette session a n=10000 (+6.45, +7.36, +11.13...).
Continuer l'entrainement jusqu'a 450k a donc reellement aide. Question
ouverte : est-ce que ca continue a progresser au-dela de 450k, ou est-ce
la que ca plafonne vraiment ? Pas encore teste.

## Extension a 600k (segments 10-12) : la progression continue, remarquablement lineaire

Meme seed (40), memes hyperparametres, `--start-segment 10` depuis
`seg_450000.pt`. 3 segments supplementaires sains (1963-2185s chacun).

Round-robin (4 specs : heuristic, vanille REINFORCE, pool 450k, pool
600k, n=10000) :

**Tete-a-tete direct 450k vs 600k** : 450k vs 600k = -2.64, 600k vs 450k
= +7.95 -- corrige, **600k bat 450k de +5.30**. Quasiment le meme ecart
que 450k vs 300k (+5.08) !

Force moyenne (contre les 3 autres) :

| | force moyenne |
|---|---|
| **Pool, 600k** | **+19.57** |
| Pool, 450k | +11.33 |
| Vanille (REINFORCE simple) | -0.81 |
| Heuristic | -30.09 |

**Classement parfaitement monotone** (600k > 450k > vanille > heuristic),
et l'ecart entre chaque palier de 150k episodes est remarquablement
stable (~+5 points a chaque fois, 300k->450k et 450k->600k). Aucun signe
de plafond -- au contraire, une progression qui ressemble a une droite.
`eval_avg` continue lui aussi de suggerer une "decline" par rapport a
300k pour ces checkpoints geants contre heuristic seul (glissement de
specialisation, cf. plus haut), donc a ne pas utiliser pour juger cette
tendance -- seul le round-robin le permet.

**A faire si on veut poursuivre** : pousser encore (750k, 900k...) pour
voir jusqu'ou la progression continue avant de vraiment plafonner.

## Bug legal_moves() a TA (2026-07-29) : corrige, impact evalue sans necessite de retrain

Trouve en jouant une partie interactive (`play_interactive.py`) : `legal_moves()`
(`coinche/game.py`) traitait TA exactement comme SA -- aucune obligation
de monter en force en suivant la couleur demandee, alors que
`regles_coinche.md` dit explicitement qu'a TA "l'ordre et les valeurs de
toutes les couleurs sont celles de l'atout" (meme logique que l'atout
demande en contrat couleur). Corrige (commit `7849f42`) : TA reutilise
desormais la logique de la branche `lead_suit == trump` (obligation de
monter si possible, sinon jouer une carte inferieure de la couleur --
jamais de defausse libre tant qu'on a cette couleur en main). SA
inchange. Verifie sur 3 scenarios + 500 parties heuristic x4 simulees
(51 a TA) sans crash.

**Impact sur les donnees d'entrainement passees, evalue avant de decider
d'un retrain** : `HeuristicPlayer._follow_card()` calcule deja lui-meme
les cartes qui battent le pli et en joue TOUJOURS une en priorite quand
c'est possible, independamment de ce que `legal_moves()` autorisait --
sa strategie respectait deja la bonne regle par sa propre logique. Les
parties heuristique-vs-heuristique (cibles d'imitation, adversaire tout
au long de l'entrainement RL) n'ont donc quasiment pas ete affectees.
Seule la policy RL pouvait techniquement echantillonner une carte
desormais illegale pendant l'exploration -- mais l'imitation avait deja
appris a reproduire le "toujours monter" de l'heuristique, donc la
probabilite sur ces coups etait deja quasi nulle avant meme le fix.
**Conclusion : pas besoin de retrain complet.** Les comparaisons
relatives de cette session restent valables (tout mesure sous la meme
regle, de bout en bout, round-robins compris).

**Note pour la prochaine session** : refaire un round-robin
d'evaluation (sanity-check sous la regle corrigee) et continuer a
pousser le pool grandissant ensemble (cf. section precedente, aucun
signe de plafond a 600k).

## Extension a 750k (segments 13-15) : la progression continue, mais ralentit nettement

Meme seed (40), memes hyperparametres, `--start-segment 13` depuis
`seg_600000.pt`. 3 segments supplementaires sains (2030-2198s chacun).

Round-robin (4 specs : heuristic, vanille REINFORCE, pool 600k, pool
750k, n=10000) :

**Tete-a-tete direct 600k vs 750k** : 600k vs 750k = -1.24, 750k vs 600k
= +2.91 -- corrige, **750k bat 600k de +2.08**. Positif, mais nettement
plus petit que les paliers precedents (+5.08 pour 300k->450k, +5.30 pour
450k->600k).

Force moyenne (contre les 3 autres) :

| | force moyenne |
|---|---|
| **Pool, 750k** | **+18.97** |
| Pool, 600k | +17.10 |
| Vanille (REINFORCE simple) | -3.00 |
| Heuristic | -33.07 |

**Toujours monotone, mais l'ecart entre paliers a ete divise par ~2.5**
(de ~+5 a ~+2). Rendements decroissants qui commencent a se voir --
premier signe (pas encore une preuve) qu'on approche d'un vrai plafond,
apres 3 paliers consecutifs de progression reelle (300k->450k->600k->750k).

**A faire si on veut trancher** : au moins un palier de plus (900k) pour
voir si l'ecart continue a se retrecir jusqu'a devenir nul, ou s'il se
stabilise a un petit gain positif residuel.

## Re-verification post-fix TA : les round-robins 300k/450k/600k tiennent (par curiosite)

Les round-robins 300k-vs-450k et 450k-vs-600k plus haut ont ete calcules
AVANT le fix du bug `legal_moves()` a TA (corrige seulement au moment de
la partie interactive, entre les extensions 600k et 750k) -- seul le
round-robin 600k-vs-750k a tourne sous le moteur corrige. Refait ici les
3 checkpoints ensemble (heuristic, vanille, 300k, 450k, 600k, n=10000)
sous le moteur corrige, par curiosite, pour voir si le fix change quoi
que ce soit.

| Comparaison | Avant le fix | Apres le fix |
|---|---|---|
| 450k bat 300k | +5.08 | +4.74 |
| 600k bat 450k | +5.30 | +6.10 |
| 600k bat 300k | (jamais mesure directement) | +10.43 |

Force moyenne (contre les 4 autres, moteur corrige) :

| | force moyenne |
|---|---|
| Pool, 600k | +17.48 |
| Pool, 450k | +9.29 |
| Pool, 300k | +6.58 |
| Vanille (REINFORCE simple) | -2.40 |
| Heuristic | -30.96 |

**Classement parfaitement monotone, ecarts quasi identiques a avant le
fix** (variation de ±10-15%, dans le bruit normal inter-runs). Confirme
directement ce que l'evaluation d'impact du bug avait predit (cf. section
precedente) : le bug etait largement inerte en pratique, la correction
ne change pas l'histoire de cette lignee d'experiences.
