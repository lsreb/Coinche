# Checkpoints RL - lignee v5 (tache auxiliaire sur le tronc partage)

Dossier separe de `rl_experiments_v4/`, cree pas pour une rupture technique
(l'etat/scoring, `imit_bignet_ent02.pt` et l'architecture `SharedTrunkActorCritic`
restent tous inchanges et reutilisables tels quels) mais parce que c'est une
nouvelle direction experimentale substantielle -- meme logique que
`rl_experiments_v4/` pour `CardNetBig` (qui ne cassait pas non plus l'etat/
scoring, juste une architecture reseau assez differente pour meriter sa
propre lignee).

Contexte (discussion du 2026-07-27, `coinche/remarques_rl.md` point 6) :
`rl_experiments_v4/` a confirme (n=4) que le tronc partage
(`SharedTrunkActorCritic`) egale REINFORCE (24.60 vs 24.42) sans le
depasser. Piste identifiee pour aller plus loin : le critic centralise a
un raccourci disponible (sa branche info-centralisee, qui voit directement
les mains des 3 autres) pour reduire sa propre perte -- son gradient sur
le tronc partage est donc dilue, puisqu'il n'est pas oblige de faire
remonter cette info dans le tronc pour ameliorer sa prediction. Une tache
auxiliaire placee sur le tronc **seul** (sans ce raccourci) n'a pas
d'autre choix que d'ameliorer le tronc lui-meme pour reduire sa perte --
un mecanisme de shaping plus direct sur la representation que l'acteur
utilise aussi.

## `SharedTrunkActorCriticAux` (`coinche/rl_agent.py`) + `SharedTrunkAuxPPOPolicy` (`train_ppo.py --architecture shared_aux`)

Sous-classe de `SharedTrunkActorCritic` (jamais modifiee en place) qui
ajoute une 3e tete, predisant le nombre d'atouts restants chez chacun des
3 autres sieges (`other_trump_counts`), depuis le tronc partage `h` SEUL
(pas la branche info-centralisee du critic -- deliberement, pour eviter
que le reseau ne "triche" en lisant la reponse dans cette branche plutot
que de l'encoder dans `h`). Une seule couche cachee (`128->32->3`, meme
profondeur que la branche info-centralisee du critic).

Nouvel hyperparametre `--aux-coef` (poids de la perte auxiliaire, MSE
normalisee par son propre ecart-type, meme logique que `return_scale`
pour `value_loss`).

**Masque SA/TA** : le "slot canonique 0" ne correspond a un vrai atout que
pour un contrat couleur (aucun a SA, les 4 couleurs a egalite a TA) --
les decisions prises pendant un contrat SA/TA sont donc exclues de la
perte auxiliaire (masque `aux_valid`), sans affecter `policy_loss`/
`value_loss` qui s'entrainent normalement dessus. Pas un biais : ca ne
fait que reduire la quantite de donnees disponibles pour la tache
auxiliaire specifiquement (proportionnellement a la frequence SA/TA dans
les encheres), et c'est plus coherent conceptuellement (la notion
"compte d'atouts" n'existe pas a SA).

Smoke-teste bout en bout (chargement depuis `imit_bignet_ent02.pt`, 200
episodes, `aux_loss` decroit deja visiblement, sauvegarde/rechargement,
`eval_policy.py --architecture shared_aux`) -- aucune erreur. **Pas
encore lance en conditions reelles.**

## Premier run reel (`aux_coef=0.5`, seed30)

Memes hyperparametres stables que `rl_experiments_v4/` (`lr=3e-5, epochs=4,
value_coef=0.2`, depuis `imit_bignet_ent02.pt`), `aux_coef=0.5` (valeur de
depart, non calibree). Entrainement sain sur toute la duree (`clip_frac`
0.8-2.4%, plage normale ; `aux_loss` decroit 0.35 -> ~0.20-0.22 ; `entropy`
stable ~0.20-0.24, pas de collapse).

**eval_avg(45000) = +15.84**, win_rate=53.3% -- nettement **sous** la
reference tronc partage sans tache auxiliaire (24.60, std=2.50, n=4) et
sous REINFORCE seul (24.42, std=1.05, n=3). Meme en tenant compte de la
variance inter-seeds deja large du tronc partage (22.09 a 27.80 sur les 4
seeds sans aux), 15.84 est en dessous du minimum observe jusqu'ici pour
cette famille d'architecture.

**Un seul seed pour l'instant** -- pas encore de conclusion definitive
(cf. le cas `lr=1e-4` dans `rl_experiments_v4/README.md`, ou un seul
seed nettement moins bon a suffi a ecarter cette piste sans creuser plus,
la config alternative etant deja confirmee a plusieurs seeds). Hypothese
la plus probable : `aux_coef=0.5` est trop eleve et la perte auxiliaire
degrade la representation partagee au lieu de l'ameliorer (l'inverse du
raisonnement initial -- un gradient plus fort sur le tronc n'est utile que
si sa direction est compatible avec ce dont l'acteur a besoin).

## `aux_coef=0.1` (seed30) : mieux, mais toujours sous la reference

Meme protocole, seul `aux_coef` change (0.5 -> 0.1). Entrainement a nouveau
sain (`clip_frac` 0.9-2.7%, `aux_loss` decroit 0.45 -> ~0.22-0.31).

**eval_avg(45000) = +19.64**, win_rate=54.1% -- mieux qu'`aux_coef=0.5`
(+15.84), confirme l'hypothese qu'un coefficient plus bas nuit moins a la
representation partagee. Reste neanmoins sous la reference tronc partage
sans tache auxiliaire (24.60, std=2.50, n=4) et sous REINFORCE seul
(24.42, n=3, std=1.05) -- la tendance suggere qu'un `aux_coef` encore plus
bas pourrait continuer a s'ameliorer plutot que d'avoir trouve un optimum.

| aux_coef | eval_avg(45000) |
|---|---|
| 0.5 | +15.84 |
| 0.1 | +19.64 |
| 0.02 | +24.19 |

## `aux_coef=0.02` (seed30) : rejoint la reference, ne la depasse pas

Tendance monotone confirmee : plus `aux_coef` baisse, plus le resultat se
rapproche de la reference tronc partage SANS tache auxiliaire (24.60,
n=4). A 0.02, l'ecart (24.19 vs 24.60) est deja dans le bruit inter-seeds
habituel de cette architecture (std=2.50). **Aucun point de la courbe ne
depasse la reference** -- l'evidence accumulee jusqu'ici va dans le sens
d'un effet neutre-a-legerement-negatif de la tache auxiliaire a toutes les
magnitudes testees, la limite `aux_coef -> 0` convergeant simplement vers
"pas de tache auxiliaire du tout".

**Lecture honnete** : rien dans ces 3 points ne supporte l'hypothese de
depart (le gradient auxiliaire, en n'ayant pas le raccourci du critic,
ameliorerait le tronc partage plus efficacement que le critic seul). Au
contraire, meme a faible poids, il semble legerement gener plutot
qu'aider -- peut-etre parce que "predire le compte d'atouts adverses"
n'est pas une representation utile a l'acteur (qui doit choisir une carte,
pas estimer une quantite), contrairement a l'intuition AlphaStar ou la
tache auxiliaire visait un signal plus directement actionnable.

## A faire

- Decider si poursuivre (ex. tester un poids encore plus proche de 0
  pour confirmer la convergence, ou changer de cible auxiliaire -- ex.
  "as restants" plutot que "atouts restants") ou considerer cette piste
  comme non concluante en l'etat, comme `lr=1e-4` en v4.
- Si poursuite : confirmer a plusieurs seeds la meilleure config avant de
  conclure quoi que ce soit, meme rigueur que partout ailleurs -- mais
  pas de sens a multiplier les seeds sur une config deja dominee par la
  reference.
