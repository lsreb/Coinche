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

## `epochs=16` a 7 seeds : la variance elevee se confirme, pas juste du petit echantillon

4 seeds de plus (23-26) pour `epochs=16` uniquement (`epochs=8` et `sans
critic` restent a n=3 -- cf. discussion sur le calcul de puissance : ca
suffit pour cette comparaison precise, la variance des deux autres etant
deja tres faible). Valeurs a n=45000 : 14.07, 15.42, 10.84, 8.29, 12.74,
16.97, 15.63.

**Moyenne ≈ 13.42, ecart-type ≈ 3.04** (n=7).

- **La moyenne n'a presque pas bouge** (13.44 a n=3 -> 13.42 a n=7) --
  c'est une estimation maintenant assez stable.
- **L'ecart-type n'a pas baisse, il a meme legerement augmente** (2.35 a
  n=3 -> 3.04 a n=7) -- ce n'etait donc pas un artefact de petit
  echantillon qui allait se resorber avec plus de seeds. `epochs=16` a
  bien une vraie dispersion large et desormais confirmee (de +8.29 a
  +16.97 selon le seed, un ecart de 8.7 points).
- La moyenne d'`epochs=16` (13.42) reste la **plus basse** des trois
  configs, et l'ecart avec `epochs=8` (1.71 pts, SE combine ~1.17, t≈1.46)
  commence a devenir suggestif sans etre encore une preuve ferme.

**Conclusion** : `epochs=16` n'est pas juste "pas prouve meilleur" comme on
le pensait a n=3 -- les donnees supplementaires penchent maintenant plutot
vers "probablement pas meilleur, et plus instable". `epochs=8` reste le
choix le plus defendable par defaut : moyenne au moins aussi bonne, et
ecart-type bien plus faible (0.39 contre 3.04) -- un resultat fiable plutot
qu'une loterie entre tres bon et mediocre.

## REINFORCE a 4 seeds par lr : "PPO bat REINFORCE" et "REINFORCE bat PPO" etaient tous les deux faux

Meme traitement que pour `epochs` : 3 seeds supplementaires (21-23) pour
REINFORCE `lr=3e-5` et `lr=1e-4` (en plus du seed original), re-evalues a
`--games 45000`.

| lr | valeurs (n=4) | moyenne | ecart-type |
|---|---|---|---|
| `3e-5` | 15.39, 15.56, 14.67, 14.98 | **15.15** | **0.40** |
| `1e-4` | 14.34, 16.89, 13.87, 15.17 | **15.07** | 1.33 |

**La conclusion "baisser le lr aide aussi REINFORCE" s'effondre** : delta de
moyennes de 0.08 (contre +4.07 sur la comparaison a un seul seed, +25.14 vs
+21.07, qui semblait nette). Les deux lr donnent en realite la meme
performance a REINFORCE.

**Et ca referme la comparaison PPO vs REINFORCE de toute la lignee v3** :
`epochs=8` (PPO, valide a 3 seeds) donne moyenne **15.13**, ecart-type
**0.39**. REINFORCE `lr=3e-5` donne moyenne **15.15**, ecart-type **0.40**.
Quasi identiques. Ni "PPO bat REINFORCE" (conclusion initiale de cette
lignee, sur un seul seed chacun) ni "REINFORCE bat PPO" (conclusion
suivante, apres avoir corrige le lr de REINFORCE, toujours sur un seul
seed) ne tenaient -- les deux algorithmes convergent essentiellement vers
la meme performance sur cette tache (~15 points au-dessus d'`heuristic`,
desormais mesure de facon fiable a -0.51 et non +7 a +8 comme au debut de
la session).

## `--no-critic-baseline` re-evalue a 45000 : la conclusion "le critic compte" s'effondre aussi, et s'inverse

Les 3 checkpoints `epochs=16` sans critic (seed20/21/22, jamais re-evalues
au-dela de n=5000) re-evalues a `--games 45000` -- meme protocole que le
reste de cette section, aucun nouvel entrainement necessaire.

| | valeurs (n=3) | moyenne | ecart-type |
|---|---|---|---|
| `epochs=16` avec critic (n=7, cf. plus haut) | 14.07, 15.42, 10.84, 8.29, 12.74, 16.97, 15.63 | 13.42 | 3.04 |
| `epochs=16` sans critic (n=3) | 14.89, 13.64, 13.80 | **14.11** | **0.68** |

**La conclusion initiale ("le critic compte", delta -8.11 sur le seed20 a
n=5000 : +27.00 avec critic vs +18.89 sans) s'effondre et s'inverse** :
`sans critic` a maintenant une moyenne legerement **superieure** (14.11 vs
13.42, ecart non significatif vu la variance du groupe avec critic) --
l'oppose du delta initial. Le "+27.00" du seed20 n'etait qu'un tirage de
mesure gonfle (sa vraie valeur, remesuree a n=45000, est 14.07). Bonus :
`sans critic` garde un ecart-type aussi faible (0.68) que les autres
configs stables de la session (`epochs=8` PPO, REINFORCE `lr=3e-5`) -- c'est
`epochs=16` **avec** critic qui est l'anomalie instable, pas le retrait du
critic.

**A ce stade, comme pour `--no-counterfactual-baseline`, aucune preuve que
le critic apporte quoi que ce soit de mesurable** sur cette tache -- les
trois tentatives de la session pour le rendre utile (pre-entrainement,
cible plus riche, presence/absence) n'ont jamais tenu une fois testees
rigoureusement.

## Critic centralise (CTDE) : `encode_full_state` (263-dim) reserve au ValueNet

Le critic a information partielle plafonne (R2~0.04-0.06, cf. section
precedente) a cause de l'information cachee (mains adverses). Teste donc un
critic **centralise** : `ValueNet` prend desormais `encode_full_state`
(`coinche/rl_agent.py`) = `encode_state` (167-dim, ce que voit l'acteur) +
les mains ACTUELLES des 3 autres sieges (3*32=96-dim) = 263-dim
(`FULL_STATE_DIM`). Seul le critic voit cette information privilegiee --
l'acteur (`CardNet`, reseau independant) continue de decider a partir des
seules 167 dimensions de `encode_state`, exactement comme avant. Disponible
uniquement parce que l'entrainement simule la donne entiere (les 4 mains
sont connues du process) ; jamais utilisable en jeu reel. Analogue a un
joueur humain qui revoit sa donne apres coup avec toute l'information
revelee pour mieux juger ses choix, sans que ca change ce qu'il pouvait
voir au moment de jouer.

Meme protocole que le reste de cette section : 3 seeds (20/21/22), memes
hyperparametres que la config `epochs=8` deja validee (`lr=3e-5, epochs=8,
100k episodes`), evalues sur les memes 45000 donnes fixees (`seed=42`).

| | valeurs (n=3) | moyenne | ecart-type |
|---|---|---|---|
| critic partiel (`encode_state`, epochs=8, reference) | 14.81, 15.56, 15.01 | 15.13 | 0.39 |
| critic centralise (`encode_full_state`, memes hyperparametres) | 16.25, 16.32, 17.69 | **16.75** | 0.81 |

**Delta de moyennes +1.63** (~11% de mieux que l'avantage de +15.13 sur
`heuristic`), et **separation complete** entre les deux groupes : le
minimum du groupe centralise (16.25) depasse le maximum du groupe partiel
(15.56) -- le signal le plus net qu'un plan a n=3 vs n=3 puisse produire
(test de permutation unilateral sur les rangs : p=0.05 exactement, la
meilleure valeur atteignable a cette taille d'echantillon). Un test de
Welch sur ces memes chiffres donne t=3.13 (df≈2.9), juste EN-DESSOUS du
seuil conventionnel a deux queues (t_crit≈3.18 a df=3) -- borderline, dans
la meme veine que le reste de cette section.

Point de vigilance : l'ecart-type du groupe centralise (0.81) est deja
environ le double de celui des configs stables de cette lignee (0.39-0.40
pour `epochs=8` PPO et REINFORCE `lr=3e-5`) -- le meme signal precurseur
qui, pour `epochs=16`, s'est revele annoncer une vraie instabilite de seed
une fois mesuree a n=7 (ecart-type 3.04). Rien ne dit que ce soit le cas
ici, mais rien ne l'exclut non plus a n=3.

**Verdict prudent** : le critic centralise semble aider (+1.63 en moyenne,
separation complete des 3+3 seeds valeurs) -- c'est, a ce stade de la
session, le seul essai sur le critic qui montre un effet dans le sens
attendu plutot que de s'effondrer sous test rigoureux. Mais n=3 vs n=3 est
la plus petite base justifiable dans la methodologie appliquee tout au
long de ce README : ce n'est pas encore le niveau de confiance des autres
conclusions tranchees ici (`epochs=8` vs REINFORCE, `epochs=16` moins
stable, absence d'effet du critic partiel). 3-4 seeds de plus de chaque
cote permettraient de confirmer -- ou de degonfler, comme tant d'autres
resultats prometteurs de cette session -- ce delta.

## Critic centralise re-teste a n=7 : le delta se degonfle et redevient non significatif

4 seeds supplementaires (23/24/25/26, meme protocole exactement), portant
le groupe critic centralise a n=7 -- meme demarche que celle qui avait
revele l'instabilite reelle d'`epochs=16` (variance sous-estimee a n=3).

| | valeurs | moyenne | ecart-type |
|---|---|---|---|
| critic centralise (n=3, seeds 20-22) | 16.25, 16.32, 17.69 | 16.75 | 0.81 |
| critic centralise (n=7, seeds 20-26) | 16.25, 16.32, 17.69, 15.89, 17.32, 13.02, 15.34 | **15.98** | **1.53** |
| critic partiel (n=3, reference) | 14.81, 15.56, 15.01 | 15.13 | 0.39 |

**Le delta se degonfle et n'est plus significatif** : moyenne 15.98 contre
15.13 pour le critic partiel (delta +0.85, contre +1.63 mesure a n=3), et
l'ecart-type du groupe centralise a quasiment double (1.53 contre 0.81) --
le seul seed25 (13.02, le plus bas des 7) aurait suffi a lui seul a casser
la separation complete observee a n=3. Un test de Welch sur ces chiffres
donne t=1.37 (df≈7.4), loin du seuil (t_crit≈2.36 a df=7) : la difference
n'est plus distinguable du bruit.

**Meme lecon que pour `epochs=16`** : un ecart-type qui parait faible a
n=3 peut n'etre qu'un echantillonnage chanceux, pas une propriete stable
de la config -- l'ecart-type ne peut pas se mesurer de facon fiable avec
si peu de seeds. **Verdict final (jusqu'a plus ample echantillon) : le
critic centralise ne montre pas d'avantage mesurable** sur le critic
partiel. C'est la quatrieme tentative de la session pour rendre le critic
utile (apres pretraining supervise, cible contre-factuelle enrichie, test
de presence/absence) et la quatrieme fois que le signal initial ne
survit pas a un echantillon plus grand.

## Ablation de la feature "points de plis engranges par camp" (`_points_so_far`)

Cette feature (2 scalaires dans `encode_state`, ajoutee sur l'intuition
qu'un joueur humain en tient compte, cf. plus haut) n'avait jamais ete
isolee des autres changements survenus en meme temps (lr, dimension
d'etat). Teste ici proprement : meme protocole partout (`epochs=8,
lr=3e-5, --no-critic-baseline` -- choisi car le critic, sous toutes ses
formes testees dans ce README, ne montre aucun effet mesurable, donc
autant retirer ce confondant en plus), 3 seeds (20/21/22) par bras,
`imit.pt`/`imit_ablated.pt` regeneres a l'identique (memes hyperparametres
de pretrain, seed 0) pour que chaque bras n'ait jamais "vu" la feature
different de l'autre des le depart. `ablate_points=True` force le bloc
`_points_so_far` a zero (acteur ET critic si non desactive) sans changer
`STATE_DIM`, cf. `coinche/rl_agent.py`/`train_ppo.py --ablate-points-so-far`.

| | valeurs (n=3) | moyenne | ecart-type |
|---|---|---|---|
| avec la feature | 17.47, 16.26, 13.44 | 15.72 | 2.07 |
| sans la feature (ablation) | 14.06, 15.69, 15.61 | 15.12 | 0.92 |

**Aucun effet mesurable** : delta de moyennes +0.60, largement dans le
bruit des deux groupes (ecarts-types 2.07 et 0.92 -- bien plus larges que
le delta lui-meme). Un test de Welch donne t=0.46, tres loin de tout
seuil de significativite. La feature ne fait ni gagner ni perdre a
l'acteur, une fois isolee de tout confondant -- coherente avec la
conclusion deja etablie sur le critic (aucune forme de critic teste dans
ce README n'a jamais montre d'effet mesurable, et cette feature avait ete
ajoutee justement dans l'idee qu'elle aiderait le critic). Elle reste dans
`encode_state` par defaut (l'intuition qui l'a motivee reste valable meme
sans preuve de gain chiffre), mais ce n'est pas elle qui explique une
quelconque difference de performance observee ailleurs dans ce README.

## Pool grandissant + PFSP, retest sur l'etat/scoring actuels (`run_growing_pool.py`)

Deja tente une fois dans la lignee v2 (ancien scoring, avant le fix regle
5, single-seed, evalue a n=3000 seulement) : `exp_growing_pool` y montrait
un pic au segment 6/10 (300k ep.) suivi d'une redescente non significative
-- conclusion tiede, jamais reconfirmee sur l'etat/scoring actuels ni avec
la rigueur (n=45000, multi-seed) etablie plus haut dans ce README.

Retest ici : seed20, 300k episodes, 6 segments de 50k (`run_growing_pool.py
--total-episodes 300000 --segment-episodes 50000 --init-load
rl_experiments_v3/imit.pt --seed-start 20`), reglages inchanges (`lr=1e-4,
entropy-beta=0.002`, `--pfsp-refresh-every 1000 --pfsp-temperature 0.1
--pfsp-ema-beta 0.98`, pool = heuristic + tous les checkpoints de segments
precedents). Duree reelle : 5603s (~1h33) pour les 6 segments (670, 892,
970, 1020, 1014, 1037s -- croissant avec la taille du pool, meme
comportement que v2).

Evalue a n=45000 contre heuristic seul (meme protocole que le reste de ce
README) :

| Checkpoint | eval_avg(45000) | win_rate |
|---|---|---|
| imit.pt | -2.10 | 49.5% |
| seg_50000 | +12.32 | 52.4% |
| seg_100000 | +15.74 | 53.1% |
| **seg_150000** | **+17.81** | **53.6%** |
| seg_200000 | +15.18 | 53.0% |
| seg_250000 | +13.87 | 52.9% |
| seg_300000 | +12.46 | 52.6% |

Meme dessin qu'en v2 : ca monte, pique a mi-parcours (seg_150000, 50% du
run), puis redescend -- ici jusqu'a quasiment retomber au niveau de
seg_50000. Le point final est **en dessous** du plateau stable obtenu par
un entrainement simple contre heuristic seul, sans pool (epochs=8 PPO ou
REINFORCE, ~15.1-15.2 sur 3-4 seeds, cf. plus haut).

**Mais `eval_avg` contre heuristic seul ne mesure que l'exploitation d'un
adversaire fixe, pas la polyvalence** -- exactement ce qu'un pool
grandissant est cense sacrifier un peu pour gagner en robustesse. Verifie
via un round-robin policy-vs-policy (`eval_matchup.py`, nouveau script :
memes `generate_fixed_deals`/`evaluate_fixed` que `eval_policy.py`, mais
oppose deux checkpoints entre eux au lieu d'un checkpoint a heuristic),
n=10000 donnes par appariement, entre `heuristic`, le checkpoint "vanille"
(`exp_ppo_lr3e-5/ppo_100000.pt`, entraine uniquement contre heuristic),
`seg_50000`, `seg_150000` (le pic vs heuristic) et `seg_300000` (le
final) :

| A \ B (siege 0/2 vs 1/3) | heuristic | vanille | seg50k | seg150k | seg300k |
|---|---|---|---|---|---|
| heuristic | -- | -11.46 | -9.42 | -13.01 | -9.29 |
| vanille | +17.63 | -- | +6.87 | -0.24 | -1.32 |
| seg50k | +14.89 | -1.70 | -- | -4.97 | -5.89 |
| seg150k | +22.46 | +5.88 | +12.54 | -- | +0.95 |
| seg300k | +15.77 | +3.51 | +10.29 | +4.56 | -- |

**L'hypothese se confirme** : `seg300k` bat `vanille` dans les deux sens
(+3.51 / -1.32) alors que `vanille` score mieux contre heuristic seul dans
cette meme passe (+17.63 vs +15.77) -- le reseau specialise a exploiter
heuristic perd en tete-a-tete contre un reseau moins specialise mais plus
polyvalent. Et `seg300k` est quasi a egalite avec `seg150k` en tete-a-tete
(+0.95 / +4.56) alors que `seg150k` score nettement mieux contre heuristic
seul (+17.81 vs +12.46) -- la "redescente" du tableau vs-heuristic ne
correspond donc **pas** a une regression de niveau general, seulement a un
deplacement de la specialisation (moins d'exploitation pure de heuristic,
sans perte de force face a d'autres styles). `seg300k` bat aussi
nettement `seg50k` dans les deux sens (+10.29 / -5.89) : la progression du
pool est reelle, pas du bruit.

Reserve : un seul seed (comme toujours dans ce README pour une premiere
passe), et les deux sens d'un meme appariement au round-robin ne sont pas
parfaitement symetriques (ex. vanille vs seg50k : +6.87 puis -1.70) --
bruit du placement des sieges sur les memes donnes, a lire comme direction
plutot que valeur exacte.

**A refaire** : 2 seeds de plus (meme config, seed21/22) pour confirmer
que ce pattern (progression reelle en polyvalence, malgre une redescente
trompeuse de `eval_avg` vs heuristic seul) se reproduit -- et si confirme,
ajouter un round-robin plus large (plus de segments,+ checkpoint REINFORCE
`exp_reinforce_lr3e-5`) pour caracteriser completement la polyvalence
gagnee. Reporte a une session ulterieure (~1h33/seed, ~7h pour 3 seeds
complets a 300k).

### Seed21 : le pattern se confirme, sur une trajectoire `eval_avg` completement differente

Seed21 (meme config exactement, `--seed-start 21`) : 5519s (~1h32).
`eval_avg` vs heuristic seul dessine cette fois une montee quasi continue
(pas de pic-puis-effondrement comme seed20) :

| Checkpoint | eval_avg(45000) | win_rate |
|---|---|---|
| seg_50000 | +13.60 | 52.7% |
| seg_100000 | +15.74 | 53.2% |
| seg_150000 | +16.90 | 53.4% |
| seg_200000 | +19.37 | 53.9% |
| seg_250000 | +22.25 | 54.4% |
| seg_300000 | +20.81 | 54.2% |

Deux seeds, deux formes tres differentes de courbe `eval_avg` vs
heuristic (pic-effondrement pour seed20, montee-leger repli pour seed21)
-- nouvelle confirmation que cette seule metrique est trop bruitee/trop
etroite (un seul adversaire fixe) pour juger un entrainement en pool.

Round-robin etendu (`eval_matchup.py`, n=10000/appariement) : heuristic,
vanille, seed20/seg_50000, seed20/seg_300000, seed21/seg_50000,
seed21/seg_300000 (6 specs, 30 appariements) :

| A \ B | heuristic | vanille | seed20/50k | seed20/300k | seed21/50k | seed21/300k |
|---|---|---|---|---|---|---|
| heuristic | -- | -11.46 | -9.42 | -9.29 | -8.59 | -16.17 |
| vanille | +17.63 | -- | +6.87 | -1.32 | +5.74 | -5.05 |
| seed20/50k | +14.89 | -1.70 | -- | -5.89 | +0.44 | -11.84 |
| seed20/300k | +15.77 | +3.51 | +10.29 | -- | +6.37 | +0.31 |
| seed21/50k | +17.91 | +1.59 | +4.35 | -1.01 | -- | -7.21 |
| seed21/300k | +23.80 | +9.86 | +16.92 | +4.19 | +13.03 | -- |

**Decouverte methodologique en marge** : la somme des deux sens d'un
meme appariement (X vs Y + Y vs X) n'est presque jamais nulle -- il y a
un biais de placement systematique d'environ +2 a +5 points en faveur de
qui occupe les sieges 0/2 (probablement lie a la rotation du donneur dans
`generate_fixed_deals`), independant du niveau reel des deux joueurs.
Comme tout le reste de ce README compare toujours le RL en 0/2 contre
heuristic en 1/3, ce biais n'a jamais fausse une seule comparaison
anterieure (applique identiquement a tous les checkpoints) -- mais dans
un round-robin ou chaque checkpoint occupe tour a tour les deux sieges,
il faut le neutraliser en moyennant les deux sens de chaque paire :
`skill(X,Y) = ((X vs Y) - (Y vs X)) / 2`.

Force moyenne de chacun contre les 5 autres, une fois ce biais corrige :

| | force moyenne (corrigee) |
|---|---|
| **seed21/seg_300000** | **+10.78** |
| seed20/seg_300000 | +4.96 |
| vanille | +2.21 |
| seed21/seg_50000 | -0.14 |
| seed20/seg_50000 | -3.31 |
| heuristic | -14.49 |

**Le pattern se confirme sur les 2 seeds** : les deux checkpoints finaux
(300k) du pool grandissant battent la vanille en force reelle, malgre des
trajectoires `eval_avg` vs heuristic totalement differentes en surface
(effondrement pour seed20, quasi-plateau pour seed21). Les deux
checkpoints precoces (50k) restent proches l'un de l'autre et sous la
vanille. Vraie variance de seed sur le niveau final atteint (seed21 net-
tement au-dessus de seed20), mais la conclusion qualitative -- le pool
grandissant produit un reseau plus fort en general que l'entrainement
simple, meme quand `eval_avg` vs heuristic seul suggere le contraire --
tient desormais sur 2 seeds independants.

**A refaire** : seed22 (dernier des 3 prevus) pour une confirmation a
n=3 ; envisager d'etendre le round-robin a REINFORCE simple
(`exp_reinforce_lr3e-5`) et a PPO en pool grandissant (`run_growing_pool.py`
ne supporte que `train.py`/REINFORCE actuellement -- jamais teste avec
`train_ppo.py`).

### Seed22 (n=3) : conclusion finale -- confirme sans exception sur les 3 seeds

Seed22 (meme config, ~1h28) : `eval_avg` vs heuristic seul dessine une
troisieme forme de courbe, encore differente des deux precedentes --
montee puis **plateau stable** autour de +20 (150k a 300k), sans
effondrement (seed20) ni repli (seed21) :

| Checkpoint | eval_avg(45000) | win_rate |
|---|---|---|
| seg_50000 | +11.85 | 52.3% |
| seg_100000 | +16.54 | 53.3% |
| seg_150000 | +20.05 | 54.0% |
| seg_200000 | +20.40 | 54.1% |
| seg_250000 | +20.36 | 54.1% |
| seg_300000 | +20.24 | 54.1% |

Trois seeds, trois trajectoires `eval_avg` qualitativement differentes
(effondrement / repli / plateau) -- confirmation supplementaire que
cette metrique seule ne suffit pas a juger un entrainement en pool.

**Verification du biais de siege (au passage)** : avant d'etendre le
round-robin, verifie que le "biais" de +2 a +5 points en faveur des
sieges 0/2 (note dans la section seed21) n'est pas un vrai biais mecanique
mais un artefact de cet echantillon precis de 10000 donnes. Fait tourner
toute la table d'un quart de tour (mains ET donneur decales ensemble,
memes donnes) : les resultats s'inversent **exactement** (`tourne(A vs B)
= -original(B vs A)`, verifie a la decimale pres sur plusieurs paires) --
preuve que le moteur traite les 4 sieges de facon parfaitement
symetrique, et que le "biais" observe est propre a cet echantillon de
donnes precis (seed=42), pas une propriete du jeu. La correction
`skill(X,Y) = ((X vs Y) - (Y vs X))/2` deja appliquee reste donc la bonne
methode.

Round-robin etendu a 8 specs (heuristic, vanille, seed20/21/22 x
early/final, n=10000/appariement -- seules les 26 nouvelles paires
impliquant seed22 ont ete rejouees, le reste reutilise les resultats
deja obtenus) :

| A \ B | heuristic | vanille | s20-50k | s20-300k | s21-50k | s21-300k | s22-50k | s22-300k |
|---|---|---|---|---|---|---|---|---|
| heuristic | -- | -11.46 | -9.42 | -9.29 | -8.59 | -16.17 | -7.54 | -17.65 |
| vanille | +17.63 | -- | +6.87 | -1.32 | +5.74 | -5.05 | +6.03 | -8.88 |
| s20-50k | +14.89 | -1.70 | -- | -5.89 | +0.44 | -11.84 | +2.24 | -14.24 |
| s20-300k | +15.77 | +3.51 | +10.29 | -- | +6.37 | +0.31 | +9.51 | -3.69 |
| s21-50k | +17.91 | +1.59 | +4.35 | -1.01 | -- | -7.21 | +2.48 | -10.43 |
| s21-300k | +23.80 | +9.86 | +16.92 | +4.19 | +13.03 | -- | +13.42 | -2.07 |
| s22-50k | +15.03 | +0.93 | +4.35 | -3.17 | +6.25 | -6.52 | -- | -9.87 |
| s22-300k | +22.15 | +9.56 | +13.48 | +6.40 | +10.26 | +5.04 | +12.58 | -- |

Classement final (force moyenne corrigee du biais de siege, contre les 7
autres) :

| | force moyenne |
|---|---|
| **seed22/300k** | **+10.45** |
| seed21/300k | +8.61 |
| seed20/300k | +3.73 |
| vanille | +0.62 |
| seed22/50k | -1.54 |
| seed21/50k | -1.84 |
| seed20/50k | -4.50 |
| heuristic | -14.81 |

**Conclusion finale (n=3, meme rigueur que le reste de ce README) :** les
3 checkpoints finaux (300k) sont au-dessus de la vanille, et les 3
checkpoints precoces (50k) sont en dessous -- sans exception. Le pool
grandissant + PFSP (`run_growing_pool.py`, REINFORCE) produit un reseau
mesurablement plus polyvalent/plus fort en general qu'un entrainement
simple contre heuristic seul -- meme si `eval_avg` contre heuristic seul,
pris isolement, peut suggerer le contraire selon le seed (cf. seed20).
C'est la premiere piste structurelle de toute la lignee v3 dont l'effet
initial ne s'est pas degonfle a l'echantillonnage complet -- au contraire,
il s'est confirme et clarifie.

**A refaire si on veut pousser plus loin** : etendre a REINFORCE simple
(`exp_reinforce_lr3e-5`, jamais inclus au round-robin) et batir un
equivalent PPO du pool grandissant (`run_growing_pool.py` ne pilote que
`train.py` actuellement).

## A refaire dans cette lignee si on veut poursuivre

- Si on veut vraiment distinguer `epochs=8` (PPO) de REINFORCE `lr=1e-4`
  (le seul groupe encore avec un ecart-type notable, 1.33) des deux options
  a variance minimale (`epochs=8` PPO, REINFORCE `lr=3e-5`), plus de seeds
  seraient necessaires -- mais l'essentiel est deja tranche : aucun des deux
  algorithmes ne domine l'autre sur cette tache.
- ~~Confirmer (ou degonfler) le delta du critic centralise~~ : fait, 4
  seeds de plus (n=7 au total) -- le delta se degonfle (+0.85, non
  significatif), meme conclusion que les 3 autres tentatives sur le
  critic. Reste ouvert seulement si on veut aussi porter le groupe
  critic partiel a n=7 pour une comparaison totalement symetrique --
  peu de raison de s'attendre a un changement de conclusion.
