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

## A faire

- Premier run complet (100k episodes, memes hyperparametres stables que
  `rl_experiments_v4/` : `lr=3e-5, epochs=4, value_coef=0.2`, depuis
  `imit_bignet_ent02.pt`) pour une premiere valeur d'`aux_coef` (a
  calibrer -- commencer prudent, pas un saut arbitraire).
- Comparer a la reference actuelle (tronc partage sans tache auxiliaire :
  24.60, std=2.50, n=4 ; REINFORCE seul : 24.42, std=1.05, n=3).
- Si prometteur : confirmer a plusieurs seeds avant de conclure, meme
  rigueur que partout ailleurs dans cette session.
