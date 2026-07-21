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

## Ablation `--no-counterfactual-baseline` (PPO)

Le critic PPO regresse en temps normal le retour CONTRE-FACTUEL
(`reward - counterfactual_reward(...)`, point 3 de `remarques_rl.md`), un
residuel faible a predire (R2~0.06 mesure plus tot). Sur le retour BRUT
(`--no-counterfactual-baseline`), le critic a une tache bien plus facile
(R2~0.28 mesure sur ce meme retour brut) -- teste pour voir si un signal de
critic plus riche se traduit en meilleure performance finale, memes
hyperparametres sinon (`lr=3e-5, epochs=8`, 50k episodes).

| | eval_avg(5000) | win_rate |
|---|---|---|
| ppo_50000 (avec contre-factuel) | +17.90 | 53.9% |
| ppo_50000 (`--no-counterfactual-baseline`) | +18.99 | 53.9% |

**Aucune difference detectable** (delta +1.09, tres sous l'IC95%~+/-6 pts,
meme win_rate). Lecture : le contre-factuel et le critic visent tous les
deux a retirer la meme composante de variance ("chance de la donne"), juste
par des voies differentes -- le contre-factuel via une info retrospective
privilegiee (rejouer la donne entiere a l'heuristique) qu'aucun critic ne
peut reconstituer depuis un etat partiel, le critic via une regression
apprise sur le retour brut. Les deux methodes semblent se substituer plutot
que se cumuler, d'ou un resultat final similaire malgre un R2 de critic tres
different entre les deux configs.

Diagnostic complementaire ajoute a cette occasion (mesure seule, ne change
rien au comportement) : `clip_frac` dans les logs `train_ppo.py`, fraction
des echantillons ou le clip PPO (`|ratio-1| > clip_eps`) est reellement actif.
Reste bas (~0.7-1.7%) tout du long a `lr=3e-5` -- le clipping intervient
rarement a ce regime de lr.

## Ablation `--epochs 16` (100k episodes) : le meilleur resultat de la session

`clip_frac` restant bas (~0.7-1.7%) a `epochs=8, lr=3e-5` suggerait que le
clipping n'intervenait quasiment jamais -- donc probablement de la marge
pour reutiliser encore plus chaque batch avant que le clip ne freine
reellement. Teste `epochs=16` (steps/episode passe de 2 a 4), sinon memes
hyperparametres, 100k episodes directement (pas de segments).

| | eval_avg(5000) | win_rate |
|---|---|---|
| **PPO epochs=16, 100k** | **+27.00** | **55.7%** |
| REINFORCE lr=3e-5, 100k | +25.14 | 55.3% |
| PPO epochs=8, 100k | +19.98 | 54.2% |

**Le premier delta de la session qui depasse nettement l'IC95%** (+7.02 vs
`epochs=8`, contre ~+/-6 pts a n=5000) -- le resultat le plus solide obtenu
jusqu'ici, pas juste suggestif. `clip_frac` monte a ~2.5-4% (contre
~0.7-1.7% a `epochs=8`), cohorent avec plus de reutilisation du batch, mais
reste modere -- pas d'effondrement d'entropie (stable 0.15-0.22 tout du
long). PPO revient au niveau de REINFORCE `lr=3e-5` (voire legerement
au-dessus, delta +1.86, dans le bruit) une fois cette marge exploitee --
confirme que le clipping avait bien encore de la place pour extraire plus
d'apprentissage par episode collecte, l'avantage propre de PPO sur
REINFORCE.

A tester ensuite si on veut pousser plus loin : `epochs=32` (en surveillant
si `clip_frac` continue de monter significativement, signe qu'on approche
la limite ou le clip commence a vraiment contraindre).

## Ablation `--no-critic-baseline` (a `epochs=16`) : le critic compte, en interaction avec les epochs

Suite a la discussion "le critic ne sert a rien, c'est le batching qui
compte" (suggeree par `--no-counterfactual-baseline` sans effet detectable,
cf. plus haut) : test direct en retirant la soustraction de la valeur
predite dans l'avantage (`--no-critic-baseline`, nouveau flag -- l'avantage
normalise devient alors un centrage sur la moyenne du batch courant, type
REINFORCE mais scalaire par batch plutot que EMA globale ; `value_net`
n'est ni appelee ni entrainee), a `epochs=16, lr=3e-5` sinon identique.

| | eval_avg(5000) | win_rate |
|---|---|---|
| PPO epochs=16, **avec** critic | +27.00 | 55.7% |
| PPO epochs=16, **sans** critic (`--no-critic-baseline`) | +18.89 | 54.1% |
| *pour reference : PPO epochs=8, avec critic* | *+19.98* | *54.2%* |

**Le critic compte bien** -- delta de -8.11 en le retirant, au-dessus de
l'IC95%~+/-6 pts (le 2e delta de la session a clairement depasser le
bruit, apres celui d'`epochs=16` lui-meme). Ca corrige/precise la
conclusion precedente : ce n'est pas "le critic ne sert a rien, seul le
batching compte" -- c'est que critic et epochs **interagissent**. Sans
critic, `epochs=16` (+18.89) retombe au niveau d'`epochs=8` **avec** critic
(+19.98, statistiquement indiscernables) : le gain qu'on avait attribue a
"plus d'epochs" seul s'evapore quasiment sans le critic.

**Interpretation** : le role du critic n'est pas d'etre precis (R2~0.06
reste faible, cf. plus haut) mais de reduire un peu le bruit
echantillon-par-echantillon de l'avantage. Avec plus d'epochs, on reutilise
les MEMES estimations d'avantage plusieurs fois -- si elles sont bruitees
(centrage sur la moyenne du batch seulement), chaque epoch supplementaire
amplifie ce bruit plutot que d'extraire du signal reel ; avec un critic
meme imparfait, cette reutilisation devient reellement profitable. Le
benefice de `--epochs` depend donc de la presence du critic, ce n'est pas
un facteur independant -- a l'inverse de ce qu'on pensait apres le test
`--no-counterfactual-baseline` (qui, lui, ne changeait que la CIBLE du
critic, pas sa presence).

## Validation multi-seeds : les deux conclusions ci-dessus ne tiennent PAS

Les ablations `epochs=16` et `--no-critic-baseline` ci-dessus reposaient
chacune sur un seul seed (20). Repete avec 2 seeds supplementaires (21, 22)
pour les 3 configs (`epochs=8` critic, `epochs=16` critic, `epochs=16` sans
critic), meme protocole d'eval (`eval_policy.py`, 5000 parties, seed=42) :

| Config | seed20 | seed21 | seed22 | **moyenne** | **ecart-type** |
|---|---|---|---|---|---|
| epochs=8 (critic) | +19.98 | +23.45 | +20.34 | **21.26** | 1.91 |
| epochs=16 (critic) | +27.00 | +23.51 | +18.55 | **23.02** | 4.25 |
| epochs=16 (sans critic) | +18.89 | +21.30 | +19.69 | **19.96** | 1.23 |

**Aucune des deux conclusions precedentes ne survit** :
- `epochs=16` vs `epochs=8` : delta de moyennes +1.76, largement dans
  l'ecart-type d'`epochs=16` lui-meme (4.25) -- pas significatif.
- critic vs sans critic (a `epochs=16`) : delta de moyennes +3.06, meme
  constat -- pas significatif.
- Pire : le classement **s'inverse** selon le seed. A seed20, `epochs=16`
  critic domine tout (+27.00). A seed22, c'est le **pire** des trois
  (+18.55, derriere meme la version sans critic). Les deltas spectaculaires
  mesures precedemment (+7.02 pour `epochs=16`, +8.11 pour le critic)
  reposaient sur ce seed20 precis, un tirage favorable pour cette
  combinaison -- pas une difference systematique.

**Lecon** : la variance seed-a-seed (ecart-type jusqu'a 4.25 pts) est du
meme ordre que les effets qu'on croyait avoir isoles sur un seul run. Les
sections `epochs=16` et `--no-critic-baseline` ci-dessus sont conservees
telles quelles (utiles comme illustration du risque), mais leurs
conclusions sont invalidees par cette validation -- a ce stade, on ne peut
pas distinguer ces 3 configs de facon fiable avec seulement 3 seeds chacune.
Le seul constat qui reste solide : les 3 configs (moyennes 20-23) battent
toutes nettement `heuristic` (+8.07) et `imit.pt` (+1.91).

## Re-evaluation a 45000 parties : une bonne partie du "bruit seed" etait du bruit de mesure

Meme 9 checkpoints, re-evalues a `--games 45000` (au lieu de 5000) pour
separer le bruit de mesure (echantillon fini de parties) du vrai bruit
d'entrainement (cf. discussion generale sur la precision d'evaluation).

**`heuristic` contre lui-meme passe de +8.07 (n=5000) a -0.51 (n=45000)** --
bien plus proche de ce qu'on attend par symetrie (0). Toute la session a
calcule ses deltas contre une reference elle-meme bruitee a n=3000-5000.

| Config | seed20 | seed21 | seed22 | moyenne | ecart-type (etait a n=5000) |
|---|---|---|---|---|---|
| epochs=8 (critic) | +14.81 | +15.56 | +15.01 | 15.13 | **0.39** (etait 1.91) |
| epochs=16 (critic) | +14.07 | +15.42 | +10.84 | 13.44 | **2.35** (etait 4.25) |
| epochs=16 (sans critic) | +14.89 | +13.64 | +13.80 | 14.11 | **0.68** (etait 1.23) |

**Confirme** : pour `epochs=8` et `sans critic`, l'ecart-type s'effondre
presque a zero -- la quasi-totalite de la variance mesuree a n=5000 etait
du bruit d'evaluation, pas une vraie difference entre seeds (les 3 seeds
de chaque groupe sont maintenant quasi indiscernables).

**Mais `epochs=16` reste nettement plus disperse** (2.35, contre 0.39 et
0.68) meme a cette precision -- ce n'est plus du bruit de mesure, c'est un
signal reel : `epochs=16` semble intrinsequement une config **plus
instable** d'un seed a l'autre que `epochs=8`, cohorent avec l'intuition
que reutiliser davantage le batch augmente le risque de "coller" au bruit
specifique de ce batch plutot que d'apprendre un signal qui generalise.

Les 3 moyennes (15.13 / 13.44 / 14.11) sont maintenant tres proches --
`epochs=16` est meme legerement la PLUS BASSE des trois (inverse de la
conclusion initiale "epochs=16 est le meilleur"), meme si ce n'est pas net
vu sa propre variance. **A ce stade, aucune des 3 configs ne se distingue
clairement des deux autres** ; le seul constat solide reste qu'elles
battent toutes nettement `heuristic` (desormais mesure a -0.51, pas +8.07).

## A refaire dans cette lignee si on veut poursuivre

- Plus de seeds encore (5-10+) si on veut vraiment distinguer `epochs=8` vs
  `epochs=16` vs presence du critic -- 3 seeds ne suffisent pas vu la
  variance observee, meme reduite a n=45000.
- Meme validation multi-seeds a faire sur `lr` (l'ablation initiale
  8e-4/1e-4/3e-5/1e-5 et la comparaison a REINFORCE reposent aussi sur un
  seul seed chacune) avant de considerer ces conclusions-la comme acquises
  -- et re-evaluer a plus grand n, vu que la reference `heuristic` elle-meme
  s'est reveler bruitee a n=3000-5000.
- Isoler proprement l'effet de la nouvelle feature d'etat : comparer PPO et
  REINFORCE avec et sans la feature, a lr et nombre d'episodes egaux, sans
  confondre avec le changement de lr ou de dimension d'etat comme c'est le
  cas dans les chiffres ci-dessus.
