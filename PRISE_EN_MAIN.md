# Prise en main — Coinche RL

Ce document explique, très concrètement, ce que fait chaque script du dépôt, comment les lancer, et comment les fonctions/méthodes s'articulent entre elles. Il s'adresse à quelqu'un qui découvre le projet et n'a lu ni le code ni `CLAUDE.md`.

Pour la référence rapide (commandes canoniques, résumé d'architecture), voir `CLAUDE.md` à la racine — ce document-ci va plus loin : une entrée par fonction/méthode, avec qui l'appelle et qui elle appelle.

## Sommaire

1. [Panorama du dépôt](#1-panorama-du-dépôt)
2. [Conventions à connaître avant de lire le code](#2-conventions-à-connaître-avant-de-lire-le-code)
3. [Parcours pas à pas (commandes)](#3-parcours-pas-à-pas-commandes)
4. [Référence détaillée, fichier par fichier](#4-référence-détaillée-fichier-par-fichier)
   - [4.1 `coinche/game.py`](#41-coinchegamepy)
   - [4.2 `coinche/players.py`](#42-coincheplayerspy)
   - [4.3 `coinche/env.py`](#43-coincheenvpy)
   - [4.4 `coinche/rl_agent.py`](#44-coincherl_agentpy)
   - [4.5 `train.py`](#45-trainpy)
   - [4.6 `train_ppo.py`](#46-train_ppopy)
   - [4.7 `pretrain.py`](#47-pretrainpy)
   - [4.8 `pretrain_value.py`](#48-pretrain_valuepy)
   - [4.9 `run_growing_pool.py`](#49-run_growing_poolpy)
   - [4.10 `run_growing_pool_ppo.py`](#410-run_growing_pool_ppopy)
   - [4.11 `eval_policy.py`](#411-eval_policypy)
   - [4.12 `eval_matchup.py`](#412-eval_matchuppy)
   - [4.13 `play_test.py`](#413-play_testpy)
   - [4.14 `play_interactive.py`](#414-play_interactivepy)
   - [4.15 `render_history.py`](#415-render_historypy)

---

## 1. Panorama du dépôt

Le projet a deux couches bien séparées :

- **Le moteur de jeu** (`coinche/game.py`, `coinche/players.py`, `coinche/env.py`) : implémente les règles de la Coinche (`regles_coinche.md`) et une IA à base de règles écrites à la main (`HeuristicPlayer`, qui suit `heuristiques.md`). Ne dépend d'aucune bibliothèque externe — fonctionne même sans `torch` installé.
- **La couche RL** (`coinche/rl_agent.py`, `train.py`, `train_ppo.py`, `pretrain.py`, `pretrain_value.py`, `run_growing_pool*.py`, `eval_*.py`) : entraîne un réseau de neurones à jouer la carte (pas les enchères, qui restent toujours gérées par la logique heuristique). Nécessite `torch` ; tout ce code se dégrade proprement (erreur explicite ou repli) si `torch` est absent.

Trois scripts « manuels » ferment la boucle : `play_test.py` (une partie isolée, sans humain), `play_interactive.py` (une partie jouée au clavier contre des bots) et `render_history.py` (transforme un historique JSON en page HTML consultable).

Aucun test automatisé n'existe dans ce dépôt (pas de `pytest`) : la vérification se fait en lançant réellement des parties et en inspectant le JSON/HTML produit, ou par les scripts d'évaluation (`eval_policy.py`/`eval_matchup.py`) qui comparent des checkpoints entre eux sur un grand nombre de donnes.

### Graphe de dépendances (qui importe quoi)

Notation : `A → B` signifie « A importe B ». Liste à plat (plutôt qu'un arbre ASCII) pour éviter toute arête ambiguë — en particulier, `players.py` et `rl_agent.py` n'ont **aucune** dépendance directe l'un vers l'autre, tous deux ne dépendent que de `game.py`.

```
coinche/game.py            (aucune dépendance interne)

coinche/players.py         → game.py
coinche/rl_agent.py        → game.py
coinche/env.py              → game.py, players.py

play_test.py                → players.py, rl_agent.py, env.py
play_interactive.py         → game.py, players.py, rl_agent.py, render_history.py
render_history.py           → game.py

pretrain.py                 → game.py, players.py, rl_agent.py
train.py                    → game.py, players.py, rl_agent.py
train_ppo.py                 → rl_agent.py, train.py
                               (build_opponent_pool / PFSPSampler / run_episode /
                                counterfactual_reward / evaluate / entropy_beta_for_episode)
pretrain_value.py            → game.py, players.py, rl_agent.py, train.py
                               (counterfactual_reward), train_ppo.py (PPOPolicy, ValueNet)

run_growing_pool.py         --(sous-process)--> train.py
run_growing_pool_ppo.py     --(sous-process)--> train_ppo.py

eval_policy.py               → game.py, players.py, rl_agent.py, train_ppo.py (classes Policy)
eval_matchup.py               → players.py, rl_agent.py, train_ppo.py, eval_policy.py
                                 (generate_fixed_deals, evaluate_fixed)
```

`run_growing_pool.py`/`run_growing_pool_ppo.py` n'appellent **jamais** directement les fonctions `main()` de `train.py`/`train_ppo.py` : ils lancent chaque segment comme un vrai sous-processus (`subprocess.run([sys.executable, 'train.py', ...])`), pour garantir qu'aucun état ne fuite d'un segment à l'autre.

---

## 2. Conventions à connaître avant de lire le code

- **Cartes** : une carte est représentée en texte par `f"{rang}{couleur}"`, ex. `"AP"` = As de Pique, `"10K"` = 10 de Carreau. Couleurs (`SUITS`, `coinche/game.py:4`) : `P`ique, `C`œur, `K`arreau, `T`rèfle. Rangs (`RANKS`) : `A, 10, K, Q, J, 9, 8, 7`.
- **Ordres de force** : `TRUMP_ORDER = [J,9,A,10,K,Q,8,7]` (à l'atout) vs `NORMAL_ORDER = [A,10,K,Q,J,9,8,7]` (couleur non-atout). Un contrat a un `trump` qui vaut soit une couleur (`P/C/K/T`), soit `'SA'` (sans atout — tout en `NORMAL_ORDER`), soit `'TA'` (tout atout — tout en `TRUMP_ORDER`).
- **Sièges** : 0=Nord, 1=Ouest, 2=Sud, 3=Est (`SEAT_LABELS` dans `coinche/players.py`, aligné avec `PLAYER_POSITIONS` dans `render_history.py`). Les sièges 0/2 sont une équipe, 1/3 l'autre équipe (partenaire = `(seat + 2) % 4`). Dans **tout** le code RL (`train.py`, `train_ppo.py`, `eval_*.py`), la policy entraînée/évaluée occupe **toujours** les sièges 0 et 2 (`RL_SEATS = (0, 2)`), l'adversaire les sièges 1 et 3.
- **Enchère (bid)** : un tuple `(level:int, trump:str, coinched:bool, capot:bool)`, ou `None` pour passer. `level` est un multiple de 10 entre 80 et 250 (250 = capot).
- **Historique de partie (`GameEngine.history`)** : un dict `{'deal_hands', 'auction', 'tricks', 'contract', 'raw_points'}` — c'est le seul canal par lequel `HeuristicPlayer`/`HumanPlayer`/le réseau de neurones observent ce qui s'est déjà passé dans la donne (personne ne « triche » en lisant directement `GameEngine.hands` des autres sièges, sauf le critic RL qui le fait *délibérément* en simulation, voir §4.4).
- **`torch is None`** : tous les modules RL testent cette condition et lèvent `SystemExit` avec un message clair plutôt que de planter avec une `ImportError` obscure, si `torch` n'est pas installé.

---

## 3. Parcours pas à pas (commandes)

Cette section donne, dans l'ordre où on les utilise réellement, les commandes concrètes. Toutes s'exécutent depuis la racine du dépôt, avec l'environnement conda `env_coinche` activé.

### 3.1 Vérifier que le moteur de jeu et le bidder heuristique tournent

```bash
python -m coinche.players
```
Exécute `coinche/players.py` en tant que script (`if __name__ == '__main__': main()`, `coinche/players.py:1194`) : distribue une main fixe à un `HeuristicPlayer` et affiche l'enchère qu'il produit. Sert de test de fumée après une modification de la logique d'enchères — aucune assertion, juste une inspection visuelle du résultat.

### 3.2 Jouer une donne isolée (sans interaction humaine) et en garder une trace JSON

```bash
python play_test.py --strategies random,random,random,random --out game_test.json
python play_test.py --strategies heuristic,heuristic,heuristic,heuristic --out game_test.json
python play_test.py --strategies heuristic,random,heuristic,random --hands-file mes_mains.json --dealer 0 --out game_test.json
```
`--strategies` prend 4 valeurs (sièges 0 à 3) parmi `random`, `heuristic` (ou `heuristic:variante`), `rl`/`agent` (⚠️ policy **aléatoire non entraînée** — `SimplePolicy`, pas un vrai checkpoint) ou `user`. `--hands-file` force une donne précise via un fichier JSON `[["AP","10P",...], [...], [...], [...]]`. C'est le point d'entrée le plus simple pour vérifier que le moteur ne plante pas après une modification de `game.py`/`players.py`.

### 3.3 Visualiser une partie sauvegardée en HTML

```bash
python3 render_history.py --history game_test.json --out game_history_viewer.html
```
Transforme le JSON produit ci-dessus en une page HTML autonome (onglets : enchères + un par pli), à ouvrir dans un navigateur. Utile pour déboguer visuellement une main litigieuse plutôt que de relire du JSON brut.

### 3.4 Jouer une donne au clavier, avec IA en soutien

```bash
python play_interactive.py --strategies heuristic,heuristic,human,heuristic
python play_interactive.py --strategies heuristic,heuristic,human,heuristic --advisor heuristic
python play_interactive.py --strategies rl:big:rl_experiments_v4/exp_growing_pool_bignet/seg_750000.pt,heuristic,human,heuristic
```
Un siège `human` (par défaut le siège 2/Sud) lit ses coups au clavier ; un `--advisor` (bot indépendant, jamais dans `engine.players`) affiche ce qu'il aurait joué à chaque décision humaine, sans influencer la vraie partie. `rl:architecture:chemin` charge un **vrai** checkpoint entraîné (contrairement à `play_test.py --strategies rl`). Génère un historique JSON + une relecture HTML à la fin, et propose de rejouer la même donne avec des bots à la place de l'humain pour comparer.

### 3.5 Pré-entraînement par imitation (point de départ de tout fine-tuning RL)

```bash
python pretrain.py --games 3000 --epochs 20 --out imit.pt
python pretrain.py --games 3000 --epochs 20 --architecture big --out imit_big.pt
```
Fait jouer 4 `HeuristicPlayer` entre eux (attaque et défense confondues), collecte à chaque décision `(état, carte choisie)`, puis entraîne un `CardNet` (ou `CardNetBig` avec `--architecture big`) par classification supervisée jusqu'à convergence. C'est le fichier `.pt` qu'on charge ensuite avec `--load` dans `train.py`/`train_ppo.py`.

**Piège à connaître pour `--architecture big`** (`CLAUDE.md`) : `pretrain.py` n'a pas de flag d'early stopping, donc la commande ci-dessus entraîne `CardNetBig` **jusqu'à pleine convergence** — et un `imit_big.pt` obtenu ainsi mesurablement **dégrade** le fine-tuning RL en aval (policy trop peakée/basse entropie pour explorer). Le checkpoint `CardNetBig` réellement utilisé dans les runs validés (`rl_experiments_v4/imit_bignet_ent02.pt`) a été obtenu par un script ad hoc, hors de `pretrain.py`, arrêté à une entropie cible (~0.20) — pas par cette commande telle quelle. `imit_big.pt` généré ici convient pour un test de fumée de l'architecture `big`, pas pour un vrai run de pool grandissant (voir §3.9).

### 3.6 Pré-entraînement du critic PPO (optionnel mais recommandé avant PPO)

```bash
python pretrain_value.py --policy imit.pt --games 20000 --epochs 60 --out value_imit.pt
```
Fait jouer `imit.pt` aux sièges 0/2 contre `HeuristicPlayer` aux sièges 1/3, calcule le retour **contre-factuel** de chaque donne (la vraie cible que PPO entraîne), et régresse un `ValueNet` dessus par MSE. À charger ensuite avec `train_ppo.py --load-value`.

### 3.7 Fine-tuning REINFORCE

```bash
python train.py --load imit.pt --episodes 100000 --lr 3e-5 --opponent heuristic --save exp1/final.pt
python train.py --load imit.pt --opponent heuristic,exp1/final.pt --pfsp --episodes 100000  # pool + PFSP
```
Entraîne la policy (partagée entre les sièges 0/2) contre `--opponent` (un adversaire fixe, une copie figée d'un ancien checkpoint, ou un pool avec tirage uniforme/PFSP). Le signal de récompense utilisé pour la mise à jour est un écart contre-factuel, pas le score brut (voir §4.5).

### 3.8 Fine-tuning PPO (alternative à REINFORCE)

```bash
python train_ppo.py --load imit.pt --load-value value_imit.pt --episodes 100000 --lr 3e-5 --opponent heuristic
```
Même tâche/état/contre-factuel que `train.py`, mécanisme d'apprentissage différent (batch + critic + clipping, voir §4.6). `--architecture` choisit entre réseaux indépendants (`small`/`big`) ou tronc partagé acteur/critic (`shared`/`shared_aux`/`shared_deep`).

### 3.9 Auto-play en pool grandissant — REINFORCE (la seule idée structurelle qui gagne de façon fiable)

```bash
python run_growing_pool.py --total-episodes 500000 --segment-episodes 50000 --init-load rl_experiments_v4/imit_bignet_ent02.pt --out-dir rl_experiments_v4/exp_growing_pool_bignet --architecture big
```
Découpe l'entraînement en segments `train.py` successifs ; chaque checkpoint produit rejoint le pool d'adversaires du segment suivant (avec PFSP). C'est l'orchestrateur, pas un entraînement direct — voir §4.9 pour les options d'élagage du pool (`--pool-keep-every`, `--pool-max-size`, `--pool-exclude`).

**`--init-load` doit être un checkpoint de la même architecture que `--architecture`** : avec `--architecture big`, `--init-load` doit pointer vers un état `CardNetBig` (ex. `rl_experiments_v4/imit_bignet_ent02.pt` ci-dessus, voir le piège de §3.5 — surtout pas le `imit_big.pt` généré tel quel par `pretrain.py --architecture big`, ni a fortiori `imit.pt`, qui est un `CardNet` et ferait échouer le chargement avec une erreur de clés manquantes/inattendues) ; avec `--architecture small` (le défaut), `--init-load` doit être un `imit.pt` classique.

### 3.10 Auto-play en pool grandissant — PPO

```bash
python run_growing_pool_ppo.py --total-episodes 300000 --segment-episodes 50000 --seed-start 20
```
Équivalent PPO de 3.9, pilote `train_ppo.py` au lieu de `train.py`.

### 3.11 Évaluer un ou plusieurs checkpoints

```bash
python eval_policy.py --games 45000 heuristic imit.pt exp_foo/final.pt
python eval_matchup.py --games 10000 heuristic imit.pt big:exp_foo/final.pt shared:exp_bar/final.pt
```
`eval_policy.py` : chaque checkpoint (mode glouton) contre `HeuristicPlayer` seul, sur exactement le même jeu de donnes fixées pour tous — comparaison équitable. `eval_matchup.py` : round-robin, chaque checkpoint contre **chaque autre** (nécessaire pour juger la polyvalence d'un checkpoint entraîné en self-play, que `eval_policy.py` seul peut sous-estimer — voir §4.12).

### 3.12 Pipeline bout-en-bout typique

Schéma simplifié (pas des commandes copiables telles quelles — les flags complets sont donnés dans les sous-sections 3.5/3.9/3.11/3.4 ci-dessus). L'étape 1 n'est **pas** une commande `pretrain.py` toute seule pour `--architecture big` — voir le piège documenté en §3.5 (`imit_big.pt` de plein-convergence dégrade le RL en aval ; le vrai point de départ, `rl_experiments_v4/imit_bignet_ent02.pt`, vient d'un arrêt anticipé ad hoc sur l'entropie, hors `pretrain.py`) :

```
[script ad hoc, arrêt à entropie cible ~0.20] → rl_experiments_v4/imit_bignet_ent02.pt
        │
        ▼
run_growing_pool.py --init-load rl_experiments_v4/imit_bignet_ent02.pt --architecture big → segments seg_*.pt
        │
        ▼
eval_matchup.py heuristic rl_experiments_v4/imit_bignet_ent02.pt seg_500000.pt ...   (vérifier la vraie progression)
        │
        ▼
play_interactive.py --strategies rl:big:seg_500000.pt,heuristic,human,heuristic   (jouer contre / avec)
```

---

## 4. Référence détaillée, fichier par fichier

### 4.1 `coinche/game.py`

Moteur de règles pur, sans dépendance à `players.py` ni à `torch` — le seul fichier que tout le reste importe transitivement. Implémente `regles_coinche.md` dans son intégralité (ordre des cartes, légalité des coups, scoring §5 y compris coinche/capot/dizaine/échelle TA).

**Constantes de module** (`SUITS`, `RANKS`, `TRUMP_ORDER`, `NORMAL_ORDER`, `TRUMP_POINTS`, `NORMAL_POINTS`, `SA_POINTS`, `game.py:4-12`) : les tables brutes de `regles_coinche.md` §3 (ordre des cartes) et §5 (valeur des cartes selon le type de contrat). Réutilisées telles quelles par `players.py`, `rl_agent.py` et `render_history.py` (import direct, pas de duplication).

#### `Card` (`game.py:14`)
Une carte = `(suit, rank)` + un `__repr__` qui produit le format texte `"AP"`. C'est cette représentation texte qui circule dans `GameEngine.history` (JSON-sérialisable) ; les objets `Card` eux-mêmes ne vivent que pendant la donne en cours (dans les mains des joueurs).

#### `Deck` (`game.py:22`)
- `__init__` : construit les 32 cartes (produit cartésien `SUITS × RANKS`).
- `shuffle()` : mélange en place via `random.shuffle`.
- `deal() -> List[List[Card]]` : mélange puis découpe en 4 paquets de 8. Appelée uniquement par `GameEngine.deal()` quand aucune main n'est fournie explicitement.

#### `Contract` (`game.py:34`)
Structure passive `(level, trump, coinched, capot)` — pas de logique, juste un conteneur. Construit par `GameEngine.run_auction()` une fois l'enchère terminée, lu par `GameEngine.play()` pour le scoring et par tout le reste (`players.py`, `rl_agent.py`, `render_history.py`) via `engine.contract`.

#### `GameEngine` (`game.py:41`)
Le cœur du moteur — possède `players` (4 objets `Player`), `dealer`, `hands`, `contract`, `taker_idx`, `history`. Une instance = une donne (pas un match complet à 3000 points).

- **`deal(hands=None)`** (`game.py:51`) : distribue les cartes (soit via `Deck().deal()`, soit en parsant des chaînes fournies, ex. `["AP","10P",...]`), appelle `player.deal(hand)` sur chaque joueur (leur permet de faire une copie de leur main de départ), leur assigne `seat` et `engine` (c'est ce `setattr` qui donne à `HeuristicPlayer`/`RLPlayer`/`HumanPlayer` accès à `self.engine.history`), et initialise `self.history`. Appelée par `run_auction()` en cas de redonne (tout le monde passe) et par tous les scripts de haut niveau (`play_test.py`, `train.py`, `eval_policy.py`, ...).
- **`_normalize_bid(bid)`** (statique, `game.py:83`) : ramène toute enchère capot à `(250, trump, coinched, True)`. Appelée par `run_auction()` sur chaque enchère brute retournée par `player.bid()`, et par `HumanPlayer.bid()` avant de valider la saisie clavier.
- **`_is_valid_bid(bid, current_best)`** (statique, `game.py:91`) : vérifie palier multiple de 10 dans `[80,250]`, règle de la coinche (doit reproduire exactement la dernière enchère, pas la surenchérir), règle de la surenchère normale (`+10` minimum). Appelée par `run_auction()` pour valider/rejeter chaque enchère, et par `HumanPlayer.bid()` pour re-demander une saisie invalide.
- **`run_auction()`** (`game.py:110`) : boucle sur les sièges à partir de `(dealer+1)%4`, appelle `player.bid(current_best)` à chacun, jusqu'à 4 passes consécutives ou une coinche (qui clôt immédiatement l'enchère). Si tout le monde passe, incrémente `dealer` et **rappelle `self.deal()` puis `self.run_auction()` récursivement** — c'est ce comportement qui piège les scripts d'évaluation naïfs (cf. `eval_policy.py::generate_fixed_deals`, §4.11, qui filtre ce cas). Fixe `self.contract`/`self.taker_idx` et alimente `history['auction']`/`history['contract']`.
- **`card_order_key(card, lead_suit, trump)`** (`game.py:158`) : renvoie un tuple `(is_trump, is_lead, -rank_idx)` comparable avec les opérateurs Python usuels — la « force » d'une carte dans le contexte d'un pli donné. Utilisée par `evaluate_trick`, `legal_moves`, et copiée conceptuellement (même logique, réimplémentée) dans `HeuristicPlayer._rank_strength` et `rl_agent._current_trick_winner_seat`.
- **`evaluate_trick(trick, trump)`** (`game.py:171`) : applique `card_order_key` à chaque carte du pli et renvoie le siège vainqueur. Appelée une fois par pli dans `play()`.
- **`legal_moves(seat, hand, trick, trump)`** (`game.py:179`) : implémente `regles_coinche.md` §4 (suivre/couper/monter/défausser), avec un traitement spécifique pour `SA` (suivre si possible, sinon défausse libre) et pour `TA` (obligation de monter dans la couleur demandée, traitée comme un atout demandé — c'est le bug corrigé le 2026-07-29, voir `CLAUDE.md`). **Ne valide jamais elle-même** ce qu'un joueur retourne : c'est chaque `Player.play_card()` qui doit l'appeler et choisir parmi son résultat. Appelée par `HeuristicPlayer.play_card()`/`RLPlayer.play_card()`/`HumanPlayer.play_card()`/`NeuralPolicy.choose_card()`/`PPOPolicy.choose_card()` — jamais par `GameEngine.play()` lui-même.
- **`card_point(card, trump)`** (`game.py:244`) : valeur en points d'une carte selon `TRUMP_POINTS`/`SA_POINTS`/`NORMAL_POINTS`. Appelée par `play()` pour le score final, et dupliquée volontairement (même table, pas d'import direct des méthodes d'instance) dans `render_history.card_point` et `rl_agent._points_so_far` (via `engine.card_point`, celle-ci *est* réutilisée).
- **`_round_dizaine(points, direction)`** (statique, `game.py:252`) : arrondit à la dizaine — `'nearest'` (standard, contrats couleur/SA), `'down'`/`'up'` (arrondi asymétrique attaque/défense propre à TA). Appelée uniquement par `play()`.
- **`play()`** (`game.py:267`) : joue les 8 plis (boucle `player.play_card()` par siège dans l'ordre, en partant du leader du pli précédent), puis calcule le score selon `regles_coinche.md` §5 en entier — capot, chute, arrondi dizaine, échelle TA (÷15 au lieu de dizaine directe), belote (créditée à l'équipe qui gagne le pli contenant la 2ᵉ des deux cartes Roi/Dame d'atout, pas forcément à qui les détenait), puis double tout via la coinche si `contract.coinched`. Retourne `(team_points: dict[0|1, int], contract)`. C'est la fonction que `CoincheEnv.step()`, `play_test.run_game()`, `train.run_episode()`/`counterfactual_reward()`/`evaluate()`, et tous les scripts d'éval appellent pour obtenir le résultat final d'une donne.

### 4.2 `coinche/players.py`

Toutes les stratégies de joueur, plus quelques helpers de formatage texte utilisés par `HumanPlayer` et `play_interactive.py`. Tout hérite de `Player` (interface commune `deal`/`bid`/`play_card`).

**Helpers de module** — `count_suit(hand, suit)` et `has_rank(hand, suit, rank)` (`players.py:8-12`) : deux prédicats triviaux réutilisés dans presque toute la logique d'enchères et de jeu de `HeuristicPlayer` (compter/tester la main). `RANK_ORDER` (`players.py:6`) est défini mais non utilisé ailleurs dans ce fichier (résidu, sans impact).

#### `Player` (`players.py:14`)
Interface de base : `deal(hand)` copie la main dans `self.hand`/`self.initial_hand` et réinitialise `self._raise_contributions` (utilisé seulement par `HeuristicPlayer`, voir plus bas) ; `bid()` renvoie `None` par défaut (passe) ; `play_card()` lève `NotImplementedError`. `GameEngine.deal()`/`play()` n'appellent jamais que ces trois méthodes — n'importe quel objet qui les implémente peut jouer une partie.

#### `RandomPlayer` (`players.py:31`)
Baseline naïve : `bid()` passe 70% du temps, sinon annonce un palier/couleur au hasard sans aucune cohérence stratégique ; `play_card()` suit la couleur demandée si possible sinon coupe si possible sinon joue n'importe quoi — pas d'usage de `legal_moves()`, donc **peut jouer un coup illégal** vis-à-vis de l'obligation de monter (c'est volontaire : sert de baseline la plus faible possible pour les tests de fumée, pas d'IA crédible).

#### `HeuristicPlayer` (`players.py:59`) — implémente `heuristiques.md`

C'est la classe la plus dense du dépôt. Elle est organisée en deux blocs, alignés sur la numérotation de `heuristiques.md`.

**Bloc enchères (heuristiques.md §1.x)**

- `_last_bid_info()` (`players.py:68`) : dernière offre *réellement faite* (pas les passes) dans l'enchère en cours, lue depuis `engine.history['auction']`. Appelée par `bid()` en tout premier pour savoir si le contexte est « je réponds à mon partenaire » ou « à un adversaire ».
- `_count_tricks(trump_suit, hi, lo, third)` (`players.py:78`) : estime le nombre de plis gagnables dans la main courante — paramétrable pour compter en couleur (`hi='A', lo='10'`), en SA ou en TA (`hi='J', lo='9', third='A'`, appelé avec `trump_suit=None`). Appelée par `bid()` (candidats couleur), `_partner_sa_remonte`/`_partner_ta_remonte` (vérifier qu'au moins 4 plis réels soutiennent une remontée) et `_can_coincher`.
- `_decrochage_ok(own_level, opp_level)` (`players.py:100`) : tolère un « petit mensonge » d'un seul palier lors d'un décrochage. Appelée uniquement dans `bid()`.
- `_my_last_bid()` / `_last_partner_bid()` (`players.py:106`, `117`) : ma dernière offre / la dernière offre de mon partenaire dans l'enchère en cours, lues dans l'historique. `_last_partner_bid` sert dans `bid()` quand un adversaire a parlé après le partenaire (je dois quand même pouvoir remonter l'offre du partenaire, pas juste décrocher tout seul).
- `_team_last_bidder_is_me()` (`players.py:131`) : vrai si, entre mon partenaire et moi, je suis le dernier à avoir réellement enchéri. Interdit le décrochage (`bid()`) quand mon partenaire ne m'a rien appris depuis ma dernière annonce.
- `_decrochage_adjusted_level(bidder_seat, bid)` (`players.py:151`) : si l'enchère examinée était elle-même un décrochage sur une offre adverse d'un palier inférieur, retourne le palier réel implicite (`level - 10`). Appelée par `_sa_known_aces` et `_combined_count` pour ne pas sur-interpréter un palier gonflé par décrochage.
- `_sa_known_aces(bidder_seat, bid)` (`players.py:170`) : table `{80:2, 90:3, 100:4}` appliquée au palier ajusté. Appelée par `_partner_sa_to_color` et `_combined_count`.
- `_my_earlier_bid_of_type(bid_type)` (`players.py:174`) : ma propre première offre SA/TA de cette enchère, si elle précède la remontée actuelle du partenaire — évite un double comptage. Appelée par `_combined_count`.
- `_partner_sa_to_color(partner_seat, partner_bid, own_color_candidates)` (`players.py:188`) : bascule sur ma propre couleur plutôt que de simplement soutenir le SA du partenaire, créditée des as qu'il a annoncés (moins un). Appelée par `bid()` juste après `_partner_sa_remonte`.
- `_partner_color_remonte(partner_bid)` (`players.py:202`) : remontée d'une annonce couleur du partenaire (soutien d'atout au premier palier, puis as extérieurs). Appelée par `bid()` quand `partner_bid[1]` est une couleur.
- `_combined_count(partner_bid, bid_type, my_count)` (`players.py:240`) : additionne les cartes maîtresses (as/valets) déjà annoncées par le partenaire (via le palier ajusté) et les miennes, sans double compte si sa remontée vient de ma propre annonce antérieure. Appelée par `_partner_sa_remonte`/`_partner_ta_remonte`.
- `_partner_sa_remonte(partner_bid)` / `_partner_ta_remonte(partner_bid)` (`players.py:253`, `275`) : remontée d'un SA/TA du partenaire — mes as/valets, puis un bonus de dix/neuf non-secs si l'équipe a ≥4 cartes maîtresses **et** ≥4 plis réels comptés (garde-fou anti-survalorisation). Appelées par `bid()`.
- **`bid(current_best)`** (`players.py:294`) : le point d'entrée — construit une liste de `candidates` (couleur §1.1, SA §1.2, TA §1.3, remontée du partenaire), choisit le maximum (à égalité de palier, préfère prolonger l'annonce du partenaire plutôt qu'un jeu solo via `_bid_key`), applique le décrochage face à un adversaire (jamais au partenaire), puis évalue la coinche (`_can_coincher`) en dernier recours si aucune surenchère ne se justifie. C'est la méthode que `GameEngine.run_auction()` appelle à chaque tour de parole ; `RLPlayer` en hérite tel quel (aucune substitution — seul `play_card` change).
- `_needed_tricks_to_coincher(level)` (`players.py:427`) : barème décroissant du nombre de plis requis en défense pour coincher (§1.4). Appelée par `_can_coincher`.
- `_defense_trump_tricks(trump)` (`players.py:433`) : évalue combien de plis mes atouts propres rapportent *en défense* (plus pessimiste qu'en attaque : 3 atouts quelconques ne comptent rien seuls). Appelée par `_can_coincher`.
- `_can_coincher(opponent_bid)` (`players.py:448`) : combine `_defense_trump_tricks` + `_count_tricks` hors-atout (couleur), ou `_count_tricks(None, ...)` (SA/TA), et compare au barème de `_needed_tricks_to_coincher`. Appelée en dernier dans `bid()`.

**Bloc jeu de la carte (heuristiques.md §2.x)**

- `_rank_strength(card, trump, lead_suit)` (`players.py:466`) : équivalent de `GameEngine.card_order_key` mais réimplémenté côté joueur (pas d'accès à `self.engine` requis) — utilisé partout dans ce bloc pour comparer deux cartes.
- `_master_ranks(trump)` (`players.py:478`) : `('J','9')` à TA, `('A','10')` sinon — la paire (carte maîtresse, carte seconde) selon le type de contrat. Appelée en tête de `play_card()` puis transmise à quasiment toutes les méthodes de ce bloc.
- `_weakest`/`_strongest(cards, trump, lead=None)` (`players.py:482`, `485`) : min/max au sens de `_rank_strength` — utilitaires appelés dans presque toutes les méthodes ci-dessous.
- `_long_suit_lead(cards, trump, master_lo)` (`players.py:488`) : dans une couleur longue sans as/valet mais avec le 10/9, choisit la pire carte de la série contiguë qui suit immédiatement le 10/9 (garde le 10/9 pour plus tard). Appelée par `_lead_attack_sa_ta`.
- `_played_cards()` / `_card_seen_before(suit, rank)` (`players.py:504`, `511`) : liste/test des cartes déjà tombées dans les plis complets, lues depuis `engine.history['tricks']`. Utilisées quasi partout dans ce bloc et par `rl_agent._played_before_this_trick` (logique équivalente, réimplémentée côté encodage d'état).
- `_trumps_remain_with_opponents(trump)` / `_trump_already_led(trump)` (`players.py:514`, `519`) : reste-t-il de l'atout ailleurs que dans ma main ? l'atout a-t-il déjà été mené au moins une fois ? Utilisées par `_lead_taker_trump`/`_lead_partner_trump` pour décider s'il faut « faire tomber » l'atout ou le garder.
- `_attaquant_principal_seat()` (`players.py:524`) : retrouve qui a **initié** le type de contrat retenu (pas forcément `engine.taker_idx`, qui n'est que l'auteur de la toute dernière enchère) — distinction nécessaire car `heuristiques.md` traite différemment le « preneur principal » et son partenaire suiveur. Appelée par `play_card()` (pour poser `is_taker`) et par `_is_coincheur_before_taker`.
- `_color_contract_coinched()` / `_coincheur_seat()` / `_is_coincheur_before_taker()` (`players.py:545`, `551`, `564`) : logique de §2.5 (partie coinchée) — identifie si le défenseur qui a lui-même coinché est assis juste avant le preneur principal, cas où la défense doit privilégier sa longue en entame absolue. Appelées par `_lead_defense`.
- `_is_confirmed_master(card, trump)` (`players.py:577`) : vrai si toutes les cartes qui dominent `card` dans sa couleur sont déjà tombées (donc elle est maîtresse même sans être l'as/le 10 littéralement). Appelée par `_lead_offsuit_master`, `_lead_defense`, `_lead_attack_sa_ta`.
- `_choose_attack_suit(trump)` (`players.py:584`) : couleur où l'attaque avait initialement le plus de cartes (basé sur `self.initial_hand`, pas la main courante). Appelée par `_lead_offsuit_master` et `_lead_coinche_defense_long_suit`.
- `_suit_to_replay_for_partner(master_hi)` (`players.py:595`) : si mon partenaire a ouvert une couleur pour la première fois et que j'ai pris avec ma carte maîtresse, retrouve cette couleur pour la rejouer en priorité. Appelée par `_lead_attack_sa_ta`.
- `_lead_offsuit_master(trump, master_hi, master_lo)` (`players.py:619`) : logique d'entame hors-atout côté attaque (contrat couleur) — carte maîtresse d'abord, puis la seconde si l'as est tombé, puis toute carte confirmée maîtresse par élimination, sinon la plus faible de la couleur d'attaque choisie. Appelée par `_lead_taker_trump`/`_lead_partner_trump`.
- `_lead_coinche_defense_long_suit(trump, master_hi, master_lo)` (`players.py:636`) : §2.5, priorité absolue à la longue (as en tête si présent) pour forcer le preneur à couper. Appelée par `_lead_defense`.
- `_lead_taker_trump(trump, master_hi, master_lo)` / `_lead_partner_trump(trump, master_hi, master_lo)` (`players.py:650`, `671`) : entame d'atout côté preneur principal vs côté partenaire suiveur — logique du valet en tête, règle du « 9 second » (le garder au premier tour d'atout), et arrêt de « faire tomber » l'atout si on sait être seul à en tenir. Appelées par `_lead_card`.
- `_lead_defense(trump, master_hi, master_lo)` (`players.py:686`) : entame côté défense — cas coinche prioritaire (`_is_coincheur_before_taker`), sinon carte maîtresse hors-atout, seconde devenue maîtresse, carte confirmée maîtresse par élimination, singleton, sinon la plus faible. Appelée par `_lead_card` et `_lead_attack_sa_ta` (cas défense à SA/TA).
- `_prefer_9_second_suit(candidates, trump, is_taker)` (`players.py:713`) : à TA, le partenaire du preneur (pas lui) préfère ouvrir dans une couleur où il a un 9 second, signal pour son partenaire. Appelée par `_lead_attack_sa_ta`.
- `_lead_attack_sa_ta(trump, master_hi, master_lo, is_taker)` (`players.py:724`) : la méthode d'entame la plus longue — priorité aux couleurs devenues maîtresses par élimination, puis rejouer la couleur d'ouverture du partenaire si j'y suis maître, puis classement couleurs longues-avec-maître / longues-sans-maître / courtes. Appelée par `_lead_card` (cas SA/TA, côté attaque).
- `_lead_card(trump, is_attacker, is_taker, master_hi, master_lo)` (`players.py:787`) : aiguillage entre les 5 méthodes d'entame ci-dessus selon `trump`/`is_attacker`/`is_taker`. Appelée uniquement par `play_card()` quand le pli est vide (`not trick`).
- `_is_absolute_master_rank(trump)` / `_partner_is_absolute_master(current_winner, trump)` (`players.py:798`, `801`) : le rang absolument imparable (valet à l'atout/TA, as à SA) ; teste si le maître actuel du pli (mon partenaire) tient cette carte imparable. Appelées par `_discard` et `_follow_card`.
- `_choose_defausse_suit()` (`players.py:813`) : couleur faible = singleton initial, ou couleur à 2 cartes initiales sans 10 ni as (`heuristiques.md` §2.3) — basé sur `self.initial_hand`, tirage aléatoire s'il y a plusieurs candidats à égalité (seul point d'aléa de `HeuristicPlayer` en dehors des enchères — d'où le piège documenté dans `eval_policy.py`, voir §4.11). Appelée par `_discard`.
- `_discard(trump, current_winner)` (`players.py:826`) : défausse dans la couleur faible choisie ; si le partenaire est déjà maître imparable du pli, se débarrasse de la plus forte carte non-maîtresse plutôt que de la plus faible (sécurise sans gâcher une carte utile). Appelée par `_follow_card` quand je ne peux ni suivre ni couper.
- `_secure_when_partner_wins(same, trump, lead, trick)` (`players.py:836`) : quand le pli est déjà gagné pour mon camp, décide laquelle de mes cartes de la couleur demandée jouer maintenant sans risque (garder la meilleure si elle n'est pas encore sécurisée, sécuriser la 2ᵉ meilleure sinon). Appelée par `_follow_card`.
- `_follow_card(trick, trump, is_attacker, master_hi, master_lo)` (`players.py:856`) : la méthode de suivi — cas « je peux suivre la couleur demandée » (avec sous-cas spécial « faire tomber l'atout » côté attaque à contrat couleur), « je dois couper/surcouper », et sinon délègue à `_discard`. Appelée uniquement par `play_card()` quand le pli n'est pas vide.
- **`play_card(seat, leader, trick, trump)`** (`players.py:919`) : calcule `is_attacker`/`is_taker` puis délègue à `_lead_card` ou `_follow_card` selon que le pli est vide, retire la carte choisie de `self.hand` et la retourne. C'est la méthode que `GameEngine.play()` appelle à chaque carte ; `RLPlayer.play_card()` s'y replie quand aucune policy n'est attachée.

#### Helpers de formatage texte (`players.py:937-1002`)
`SEAT_LABELS`, `SUIT_SYMBOLS`, `_trump_label`, `_fmt_card`, `_fmt_hand`, `_fmt_bid`, `_parse_bid_str`, `_find_card`, `_card_from_repr` — conversions triviales texte ↔ `Card`/enchère pour l'affichage et la saisie clavier. N'existent que pour `HumanPlayer` (et `play_interactive.py` qui importe `SEAT_LABELS`) ; aucune logique de jeu ne dépend d'eux.

#### `HumanPlayer` (`players.py:1005`)
Joueur piloté au clavier. `deal(hand)` affiche « nouvelle donne » après la première (cas de redonne). `_advisor_hint(kind, ...)` (`players.py:1025`) interroge un `Player` indépendant (`self.advisor`, jamais inséré dans `engine.players`) sur ce qu'il aurait choisi, en resynchronisant une **copie** de la main à chaque appel pour ne jamais muter la vraie main. `show_new_tricks()`/`_print_trick_recap()` (`players.py:1058`, `1050`) rattrapent l'affichage des plis terminés depuis le dernier appel (nécessaire car ce joueur peut ne pas être le dernier à jouer dans un pli). `bid()`/`play_card()` (`players.py:1075`, `1119`) affichent le contexte courant (enchères passées / état du pli, main triée) puis lisent stdin en boucle jusqu'à une saisie valide, en réutilisant `engine._is_valid_bid`/`engine.legal_moves` pour la légalité. Appelées uniquement par `GameEngine.run_auction()`/`play()` comme n'importe quel autre `Player`.

#### `RLPlayer(HeuristicPlayer)` (`players.py:1158`)
Hérite `bid()` de `HeuristicPlayer` sans modification (les enchères restent toujours heuristiques). Ne redéfinit que `play_card()` : délègue à `self.policy.choose_card(...)` si une policy est attachée (interface commune à `SimplePolicy`, `NeuralPolicy`, `PPOPolicy`, `SharedTrunkPPOPolicy`, `SharedTrunkAuxPPOPolicy` — toutes définies dans `rl_agent.py`/`train_ppo.py`), sinon retombe sur le jeu heuristique hérité. C'est le siège utilisé aux positions 0/2 dans `train.py`/`train_ppo.py`/`eval_policy.py`/`eval_matchup.py`.

#### `create_player(strategy, name)` (`players.py:1175`)
Fabrique à partir d'une chaîne (`'random'`, `'human'`, `'heuristic'`/`'heuristic:variante'`/`'user'`, `'rl'`/`'agent'`, sinon `RandomPlayer` par défaut). Utilisée par `play_test.py::build_players` et `play_interactive.py::build_one_player` — **pas** par `train.py`/`eval_policy.py`, qui construisent leurs `RLPlayer` directement avec une vraie policy chargée.

### 4.3 `coinche/env.py`

#### `CoincheEnv` (`env.py:5`)
Enveloppe minimaliste type Gym autour d'une seule donne (pas un match à 3000 points). `__init__(players=None, dealer=0, agent_seat=0)` construit 4 `RandomPlayer` par défaut si aucun joueur n'est fourni, et un `GameEngine` interne. `reset(hands=None)` (`env.py:15`) appelle `engine.deal()` puis `engine.run_auction()`, renvoie une observation minimale (`_get_obs`, `env.py:21` — juste la main courante, pas le pli). `step(action)` (`env.py:28`) ignore en réalité l'`action` passée (chaque joueur décide déjà lui-même via `play_card`) et appelle simplement `engine.play()` pour dérouler les 8 plis d'un coup, renvoyant `reward = team_points[agent_seat % 2]`. Utilisé uniquement par `play_test.py::run_game` — ni `train.py` ni `train_ppo.py` ne passent par cette classe (ils appellent `GameEngine` directement dans `run_episode`), car son modèle « une action = une décision RL » ne correspond pas à leur besoin de rejouer un contre-factuel avec les 4 sièges contrôlés par le même adversaire.

### 4.4 `coinche/rl_agent.py`

Tout ce qui transforme un état de jeu en vecteur numérique, et tous les réseaux de neurones. Se dégrade en `torch = None` si `torch` n'est pas installé (toutes les classes `nn.Module` sont alors indéfinies, protégées par `if torch is not None:`).

**Fonctions d'encodage bas niveau**

- `_canonical_slots(trump)` (`rl_agent.py:14`) : calcule 4 « slots » de couleur canoniques, invariants par rotation de l'atout réel — le slot 0 est toujours l'atout (contrat couleur) ou toute couleur (TA), avec l'ordre de rang associé (`TRUMP_ORDER`/`NORMAL_ORDER`). C'est la fonction pivot : tout le reste du fichier (et `train_ppo.py`) l'appelle pour ne jamais raisonner en couleur physique absolue — le réseau n'a besoin d'apprendre le concept « meilleure carte d'atout » qu'une seule fois, pas 4 fois (une par couleur physique).
- `_slot_index(suit, rank, suits, orders)` (`rl_agent.py:36`) : position `0..31` d'une carte donnée dans l'espace canonique — utilisée pour construire les vecteurs multi-hot et pour décoder l'action choisie par le réseau (indice → carte réelle) dans `NeuralPolicy.choose_card`/`PPOPolicy.choose_card`/`pretrain.RecordingHeuristicPlayer`.
- `_relative_multi_hot(cards, suits, orders)` (`rl_agent.py:42`) : vecteur 32-dim, 1.0 aux positions occupées par `cards`. Bloc de base réutilisé pour encoder la main, les cartes déjà jouées, le pli en cours, et la main initiale.
- `_one_hot(index, size)` (`rl_agent.py:49`) : one-hot générique (couleur menée, position dans le pli, type d'atout).
- `_played_before_this_trick(player, suits, orders)` (`rl_agent.py:56`) : équivalent (réimplémenté, pas appelé) de `HeuristicPlayer._played_cards`, mais retourne directement un vecteur multi-hot canonique plutôt qu'une liste de chaînes.
- `_slot_counts(vec32)` (`rl_agent.py:71`) : somme par bloc de 8 d'un vecteur 32-dim — utilisée pour dériver `length_vec` (longueur de couleur initiale) et `unknown_vec` (cartes encore inconnues) dans `encode_state`.
- `_record_void_from_trick(seat_suit_pairs, void, suits)` / `_void_vec(player, trick, suits)` (`rl_agent.py:77`, `93`) : détecte les renonces certaines (un siège qui n'a pas fourni la couleur demandée alors qu'il le devait) et produit un vecteur 12-dim (3 sièges × 4 slots). `_void_vec` parcourt l'historique des plis complets **et** le pli en cours.
- `_auction_signals_vec(player, suits)` (`rl_agent.py:119`) : pour chacun des 3 autres sièges, a-t-il annoncé/remonté chaque couleur (12-dim) ou SA/TA (6-dim) à un moment de l'enchère, même si le contrat final a fini ailleurs — 18-dim au total.
- `_led_suit_vec(player, trick, suits)` (`rl_agent.py:154`) : pour chacun des 3 autres sièges, a-t-il déjà mené un pli dans chaque couleur (12-dim) — distinct d'un simple comptage de cartes tombées, capture une préférence de couleur par siège précis (analogue numérique de `HeuristicPlayer._suit_to_replay_for_partner`).
- `_points_so_far(player, trump, engine)` (`rl_agent.py:184`) : points de cartes déjà engrangés par mon camp / l'adversaire dans les plis complets, normalisés par 162. Appelle `engine.card_point` (réutilise directement `GameEngine`, pas de duplication de la table de points ici).
- `_current_trick_winner_seat(trick, trump, engine)` (`rl_agent.py:209`) : siège actuellement maître du pli en cours, en réutilisant `engine.card_order_key` — utilisée pour dériver `partner_winning` dans `encode_state`.

**`encode_state(player, trick, trump, ablate_points=False)`** (`rl_agent.py:223`, `STATE_DIM = 167`) : assemble tous les blocs ci-dessus en un seul vecteur fixe — c'est **exactement** ce qu'un joueur réel peut savoir (jamais les mains des autres). Appelée par `NeuralPolicy.choose_card`, `PPOPolicy.choose_card`, `SharedTrunkPPOPolicy.choose_card`, `SharedTrunkAuxPPOPolicy.choose_card`, `pretrain.RecordingHeuristicPlayer.play_card` (collecte du dataset d'imitation), et en interne par `encode_full_state`. `ablate_points` est un flag d'étude d'ablation (force `points_vec` à zéro sans changer `STATE_DIM`) propagé depuis `pretrain.py --ablate-points-so-far` / `train_ppo.py --ablate-points-so-far`.

- `_other_hands_vec(player, suits, orders)` (`rl_agent.py:310`) : mains **actuelles** des 3 autres sièges — information privilégiée, lue directement dans `engine.players[other_seat].hand` (accès délibérément « triché », possible seulement parce qu'on simule la donne entière en entraînement). Appelée uniquement par `encode_full_state`.
- `_other_trump_counts(player, suits)` / `other_trump_counts(player, trump)` (`rl_agent.py:326`, `345`) : nombre d'atouts actuellement en main des 3 autres sièges — cible de la tête auxiliaire de `SharedTrunkActorCriticAux`. `other_trump_counts` est la version publique (calcule `suits` elle-même), appelée par `SharedTrunkAuxPPOPolicy.choose_card` dans `train_ppo.py`.
- **`encode_full_state(player, trick, trump, ablate_points=False)`** (`rl_agent.py:352`, `FULL_STATE_DIM = 263`) : `encode_state(...)` + `_other_hands_vec(...)` — l'état **centralisé** que seul le critic voit (jamais l'acteur). Appelée par `PPOPolicy.choose_card`/`SharedTrunkPPOPolicy.choose_card`/`SharedTrunkAuxPPOPolicy.choose_card` (dans `train_ppo.py`) et par `pretrain_value.collect_dataset`.

**`SimplePolicy`** (`rl_agent.py:369`) : politique aléatoire parmi les coups légaux, sans `torch`. C'est le repli utilisé par `play_test.py`/`play_interactive.py` pour la stratégie `'rl'`/`'agent'` **quand aucun checkpoint n'est chargé** — sert aussi de référence de progression (« la policy entraînée bat-elle au moins l'aléatoire ? »).

**Réseaux (uniquement définis si `torch is not None`)**

- `CardNet` (`rl_agent.py:381`) : 1 couche cachée (167→128, ReLU) + 1 sortie (128→32 logits, un par slot canonique). L'architecture de référence, historiquement validée — c'est celle que produit `pretrain.py` par défaut.
- `CardNetBig` (`rl_agent.py:391`) : 4 couches (167→128→128→64→32), Pre-LN LayerNorm avant chaque couche linéaire + GELU. Teste si une seule couche cachée est un plafond de capacité pour `CardNet`. `forward` unique (pas de tête séparée) — les méthodes `__init__`/`forward` suivent le même schéma mécanique que `CardNet`, juste avec 4 couches au lieu de 2.
- `SharedTrunkActorCritic` (`rl_agent.py:418`) : tronc partagé acteur/critic (167→128→128, mêmes noms de paramètres que les 2 premières couches de `CardNetBig`, exprès pour pouvoir reprendre un `imit_bignet*.pt` existant). Méthodes : `_trunk(x)` (les 2 couches partagées), `forward_actor(x)` (trunk + tête acteur privée 128→64→32), `forward(x)` (alias de `forward_actor`, pour que `NeuralPolicy`/`make_opponent_factory` de `train.py` puisse charger ce réseau comme adversaire figé sans changement de code), `forward_critic(x_full)` (découpe `x_full` en état visible + info centralisée 96-dim, passe l'état visible par le tronc partagé, l'info centralisée par une petite branche dédiée `fc_central`, concatène, puis tête critic privée), `load_actor_from_cardnetbig(path)` (remappe les clés d'un checkpoint `CardNetBig` existant vers ce réseau — seule la partie critic démarre aléatoire).
- `SharedTrunkActorCriticAux(SharedTrunkActorCritic)` (`rl_agent.py:520`) : ajoute `forward_aux(x)`, une 3ᵉ tête qui prédit le nombre d'atouts des 3 autres sièges **depuis le tronc partagé seul** (jamais depuis la branche centralisée du critic — choix délibéré pour empêcher le réseau de « tricher » en lisant la réponse dans cette branche plutôt que de l'encoder dans le tronc, ce qui viderait la tâche auxiliaire de son intérêt).
- `SharedTrunkActorCriticDeep` (`rl_agent.py:556`) : rééquilibrage — tronc à 3 couches (167→128→128→64, couvre exactement les 3 premières couches de `CardNetBig`), tête acteur à 1 seule couche (64→32, la dernière de `CardNetBig`). Classe indépendante (pas une sous-classe de `SharedTrunkActorCritic`, la forme du tronc diffère trop). Mêmes méthodes `_trunk`/`forward_actor`/`forward`/`forward_critic`/`load_actor_from_cardnetbig` que `SharedTrunkActorCritic`, adaptées à 3 couches de tronc.

**`NeuralPolicy`** (`rl_agent.py:653`) — la policy REINFORCE, utilisée par `train.py` et comme adversaire figé chargeable (`make_opponent_factory`) :
- `__init__(device, lr, baseline_beta, entropy_beta, net_cls=CardNet)` : construit `net_cls()` + un optimiseur Adam, initialise les buffers de trajectoire.
- `choose_card(player, legal, leader, trick, trump)` (`rl_agent.py:671`) : encode l'état (`encode_state`), masque les logits des coups illégaux à `-inf`, échantillonne (`record=True`, mode entraînement — mémorise log-prob et entropie) ou prend l'argmax (`record=False`, mode évaluation gloutonne), puis décode l'indice choisi en `Card` réelle via `suits`/`orders`. Appelée par `RLPlayer.play_card`.
- `update(reward)` (`rl_agent.py:692`) : une mise à jour REINFORCE par donne — `loss = -(somme des log-probs de la trajectoire) * avantage - entropy_beta * entropie`, où `avantage = reward - baseline` (baseline = moyenne mobile globale). Appelée par `train.py::main` après chaque épisode (avec le reward *contre-factuel*, pas le reward brut — voir §4.5).
- `save(path)` / `load(path)` (`rl_agent.py:711`, `714`) : `save` écrit juste le `state_dict` du réseau ; `load` accepte aussi bien ce format qu'un checkpoint PPO natif (dict à clé `'policy'`) — nécessaire pour que `make_opponent_factory` (train.py) puisse charger un checkpoint produit par `train_ppo.py` comme adversaire figé dans un pool.

### 4.5 `train.py`

Entraînement REINFORCE. Importe `GameEngine` (game.py), `RLPlayer`/`HeuristicPlayer` (players.py), `NeuralPolicy`/`CardNet`/`CardNetBig` (rl_agent.py). `RL_SEATS = (0, 2)` (`train.py:36`, documentation, non utilisé en dur ailleurs que par convention dans le code qui suit).

- `_default_opponent(name)` (`train.py:39`) : construit un `HeuristicPlayer` — la factory par défaut de tout le fichier.
- **`make_opponent_factory(token, net_cls=CardNet)`** (`train.py:43`) : convertit un token `--opponent` (`'heuristic'` ou un chemin de checkpoint) en fonction `name -> Player`. Pour un chemin, charge un `NeuralPolicy(net_cls=net_cls)` figé (`record=False`) et retourne un `lambda` qui construit un `RLPlayer` autour de cette policy partagée (même objet réutilisé pour les 2 appels de la factory — donc un seul réseau chargé en mémoire même si l'adversaire occupe 2 sièges). Appelée par `build_opponent_pool` et réutilisée telle quelle par `train_ppo.py` (import direct, pas de duplication).
- **`build_opponent_pool(opponent_arg, net_cls=CardNet)`** (`train.py:61`) : parse `--opponent` (`"token1,token2,..."`) en `[(token, factory), ...]`. Appelée par `main()` (train.py et train_ppo.py) et par `run_growing_pool.py`/`run_growing_pool_ppo.py` indirectement (ils construisent la chaîne `--opponent` passée en sous-processus, pas cette fonction directement).
- **`PFSPSampler`** (`train.py:70`) : tirage de l'adversaire biaisé vers le plus coriace (Prioritized Fictitious Self-Play, façon AlphaStar). `__init__(pool, refresh_every, temperature, ema_beta, explore_eps)` initialise un win-rate EMA à 0.5 pour chaque adversaire. `_refresh_weights()` (`train.py:109`) recalcule un softmax sur `1 - win_rate` (température réglable), mélangé avec un plancher d'exploration uniforme `explore_eps`. `choose(global_ep)` (`train.py:117`) rafraîchit les poids tous les `refresh_every` épisodes puis tire un adversaire pondéré. `record_outcome(name, reward)` (`train.py:128`) met à jour l'EMA de win-rate après chaque épisode réellement joué contre cet adversaire. `summary()` (`train.py:132`) formate un résumé texte (`"nom:Nx(wr=0.xx)"`) — c'est cette ligne, imprimée en fin d'entraînement (`"PFSP picks total: ..."`), que `run_growing_pool.py::_parse_final_winrates` reparse pour élaguer le pool par faiblesse.
- **`run_episode(policy, dealer, opponent_factory=_default_opponent)`** (`train.py:138`) : joue une donne complète, policy aux sièges 0/2, `opponent_factory` aux sièges 1/3. Retourne `(reward, deal_hands)` — `reward = team_points[0] - team_points[1]`. Appelée par `main()` (train.py et train_ppo.py) et par `evaluate()`.
- **`counterfactual_reward(hands, dealer, opponent_factory=_default_opponent)`** (`train.py:155`) : rejoue **exactement les mêmes mains** avec `opponent_factory` aux 4 sièges (pas de policy) — le signal de référence « qu'aurait fait cet adversaire à ma place, sur les mêmes cartes ». Doit toujours recevoir le même `opponent_factory` que le `run_episode` qui a produit `hands` (sinon on compare deux adversaires différents). Appelée par `main()` (les deux fichiers d'entraînement) et par `pretrain_value.collect_dataset`.
- **`evaluate(policy, n_games, start_dealer=0, opponent_factory=_default_opponent)`** (`train.py:175`) : bascule `policy.record` à `False`, joue `n_games` donnes gloutonnes via `run_episode`, restaure `policy.record`, retourne `(avg, win_rate)`. Appelée périodiquement par `main()` (dans `train.py` et `train_ppo.py`) pour logger la progression pendant l'entraînement — toujours contre `heuristic` par défaut, même en entraînement de pool, pour un suivi comparable d'un run à l'autre. C'est un suivi *pendant* l'entraînement, sur des donnes tirées au hasard à chaque appel ; l'évaluation finale « sérieuse » d'un checkpoint utilise plutôt `eval_policy.py`/`eval_matchup.py`, qui ont leur propre `evaluate_fixed` à donnes fixées et partagées entre checkpoints (voir §4.11-4.12).
- `entropy_beta_for_episode(base_beta, ep, decay)` (`train.py:196`) : calcule le coefficient d'entropie effectif selon le schéma choisi (`'none'`, `'invsqrt'`, `'inv'`). Appelée à chaque épisode par `main()` (les deux fichiers).
- **`main()`** (`train.py:209`) : parse les arguments CLI, construit le pool d'adversaires + éventuellement un `PFSPSampler`, charge `--load` si fourni, puis boucle sur `--episodes` : choisit un adversaire (pool uniforme ou PFSP), joue l'épisode (`run_episode`), calcule le reward d'entraînement (brut ou contre-factuel selon `--no-counterfactual-baseline`), appelle `policy.update(training_reward)`, logge/évalue/checkpointe périodiquement. C'est le point d'entrée exécuté par `python train.py ...` **et** celui lancé en sous-processus par `run_growing_pool.py`.

### 4.6 `train_ppo.py`

Entraînement PPO — même tâche que `train.py` (même état, mêmes sièges, même contre-factuel), mécanisme différent : batch de plusieurs donnes, critic par état, plusieurs passes clippées. Réimporte directement `build_opponent_pool`, `PFSPSampler`, `run_episode`, `counterfactual_reward`, `evaluate`, `entropy_beta_for_episode` depuis `train.py` (aucune duplication de cette logique).

- **`ValueNet`** (`train_ppo.py:73`) : critic **centralisé** (prend `encode_full_state`, `FULL_STATE_DIM`) et à tronc **indépendant** de la policy (pas de partage avec `CardNet` — un tronc partagé hérité de `imit.pt` s'est révélé incapable de séparer attaque/défense après 50k épisodes, cf. `remarques_rl.md`). 1 couche cachée (128) + 1 sortie scalaire. `forward(x)` : ReLU puis tête valeur.
- **`PPOPolicy`** (`train_ppo.py:102`) : implémente l'interface `choose_card`/`record`/`save`/`load` attendue par `RLPlayer`/`run_episode`/`evaluate`, plus `end_episode`/`update_batch` spécifiques à PPO.
  - `__init__` : `policy_net` (`CardNet`/`CardNetBig` selon `policy_net_cls`) + `value_net` (`ValueNet`) dans un **même** optimiseur Adam (un seul `.step()` met à jour les deux réseaux, qui restent néanmoins deux troncs séparés).
  - `choose_card(...)` (`train_ppo.py:133`) : encode l'état visible (acteur) et l'état complet (critic, `encode_full_state`), échantillonne l'action, et **mémorise** dans `self._traj` `(état, état_complet, action, log_prob, valeur, masque)` — contrairement à `NeuralPolicy`, rien n'est mis à jour immédiatement.
  - `end_episode(reward)` (`train_ppo.py:161`) : archive `(trajectoire, reward)` dans `self._episodes` (pas de mise à jour). Appelée par `main()` à la place de `.update()`.
  - `update_batch()` (`train_ppo.py:171`) : concatène toutes les trajectoires accumulées depuis le dernier appel, calcule l'avantage (`retour - valeur_prédite`, normalisé) sauf si `no_critic_baseline` (alors avantage = retour centré/normalisé sur le batch, `value_net` ni appelée ni entraînée), puis fait `self.epochs` passes en minibatches avec l'objectif PPO clippé (`clip_eps`) + perte de valeur MSE (`value_coef`) + bonus d'entropie (`entropy_coef`). Retourne `(policy_loss, value_loss, entropy, clip_frac)` moyens. Appelée par `main()` tous les `--batch-episodes` épisodes.
  - `save(path)`/`load(path)`/`load_value(path)` (`train_ppo.py:263`, `279`, `301`) : `save` écrit un dict `{'policy':, 'value':}` (**pas** l'état de l'optimiseur — une reprise repart avec un Adam « à froid »). `load` accepte un checkpoint PPO natif, un checkpoint REINFORCE brut (`CardNet`, même forme), ou un ancien format à tronc partagé (`policy_head`/`value_head`, rétrocompatibilité). `load_value` charge `value_net` séparément (checkpoint PPO natif ou sortie de `pretrain_value.py`).
- **`SharedTrunkPPOPolicy`** (`train_ppo.py:308`) : même interface que `PPOPolicy`, mais un seul réseau (`net_cls=SharedTrunkActorCritic` par défaut, ou `SharedTrunkActorCriticDeep`) avec un seul optimiseur — le gradient de `value_loss` traverse donc aussi le tronc partagé avec l'acteur (contrairement à `PPOPolicy`, où le tronc de `value_net` est totalement indépendant). Mêmes méthodes `choose_card`/`end_episode`/`update_batch`/`save`/`load`, adaptées pour appeler `net.forward_actor`/`net.forward_critic` au lieu de deux réseaux séparés. `load` (`train_ppo.py:450`) détecte automatiquement s'il reçoit un checkpoint natif de cette architecture ou un `CardNetBig` brut (auquel cas `net.load_actor_from_cardnetbig` est utilisée, partie critic initialisée aléatoirement).
- **`SharedTrunkAuxPPOPolicy(SharedTrunkPPOPolicy)`** (`train_ppo.py:466`) : ajoute la tâche auxiliaire (prédire le nombre d'atouts des 3 autres sièges). `choose_card` calcule en plus `aux_target` (`other_trump_counts`) et `aux_valid` (`False` à SA/TA, où la notion de « slot atout » n'a pas de sens) à chaque décision. `update_batch()` (`train_ppo.py:546`) ajoute une perte auxiliaire MSE (normalisée par son propre écart-type, `aux_coef`) calculée uniquement sur les décisions valides, et retourne un 5-tuple `(policy_loss, value_loss, aux_loss, entropy, clip_frac)`.
- **`main()`** (`train_ppo.py:625`) : parse les arguments, choisit la classe de policy selon `--architecture` (`small`/`big` → `PPOPolicy` ; `shared`/`shared_aux`/`shared_deep` → la variante à tronc partagé correspondante), charge `--load`/`--load-value`, puis boucle sur `--episodes` : joue un épisode (`run_episode`), calcule le reward contre-factuel, appelle `policy.end_episode(...)`, déclenche `policy.update_batch()` tous les `--batch-episodes`, logge/évalue/checkpointe périodiquement. C'est le point d'entrée exécuté directement **et** celui lancé en sous-processus par `run_growing_pool_ppo.py`.

### 4.7 `pretrain.py`

Pré-entraînement supervisé de `CardNet`/`CardNetBig` par imitation de `HeuristicPlayer` — le point de départ standard avant tout fine-tuning RL.

- **`RecordingHeuristicPlayer(HeuristicPlayer)`** (`pretrain.py:27`) : rejoue exactement la logique de `HeuristicPlayer` (aucune duplication) ; `play_card(...)` (`pretrain.py:36`) encode l'état (`encode_state`) et les coups légaux **avant** d'appeler `super().play_card(...)`, puis enregistre `(state, action_idx, legal_idx)` dans `self.dataset`, une liste partagée entre les 4 sièges d'une même donne.
- `collect_dataset(n_games, seed=None, ablate_points=False)` (`pretrain.py:49`) : simule `n_games` donnes 4×`RecordingHeuristicPlayer` partageant le même `dataset`, retourne la liste d'exemples collectés. Appelée par `main()`.
- `train_supervised(dataset, epochs, lr, batch_size, device, net_cls=CardNet)` (`pretrain.py:66`) : boucle d'entraînement classique — masque les logits des coups illégaux à `-inf` avant la cross-entropy (mêmes masques que `NeuralPolicy.choose_card` en inférence), affiche loss/accuracy par epoch. Appelée par `main()`.
- **`main()`** (`pretrain.py:98`) : `--games`/`--epochs`/`--architecture` (`small`=`CardNet`/`big`=`CardNetBig`)/`--ablate-points-so-far`, orchestre `collect_dataset` → `train_supervised` → `torch.save(net.state_dict(), args.out)`. C'est ce fichier `.pt` de sortie (`imit.pt` par défaut) que `train.py --load`/`train_ppo.py --load` consomment ensuite.

### 4.8 `pretrain_value.py`

Pré-entraînement supervisé de `ValueNet` (le critic de `train_ppo.py`) — régresse la cible **contre-factuelle** exacte que PPO entraîne réellement (pas le score brut : une première version sur le score brut avait un bon R² hors-ligne mais dégradait la performance PPO finale, voir docstring du module et `remarques_rl.md`).

- `collect_dataset(policy, n_games, seed=None)` (`pretrain_value.py:45`) : instrumente temporairement `policy.choose_card` (monkey-patch local, restauré à la fin) pour capturer `encode_full_state` à chaque décision, simule `n_games` donnes avec `policy` aux sièges 0/2 contre `HeuristicPlayer` aux sièges 1/3 (même schéma que `train.run_episode`), calcule pour chaque donne `target = reward - counterfactual_reward(hands, dealer)` (import direct de `train.counterfactual_reward`), et associe cette cible unique à tous les états rencontrés dans la donne. Appelée par `main()`.
- `train_supervised(dataset, epochs, lr, batch_size, device)` (`pretrain_value.py:87`) : régression MSE classique, affiche MSE et R² par epoch. Appelée par `main()`.
- **`main()`** (`pretrain_value.py:117`) : `--policy` (le checkpoint chargé aux sièges 0/2 pour la collecte, `imit.pt` par défaut via `PPOPolicy`), `--games`/`--epochs` (nettement plus élevés que `pretrain.py` — le résiduel après soustraction du contre-factuel est un signal beaucoup plus faible), orchestre `collect_dataset` → `train_supervised` → sauvegarde. Sortie (`value_imit.pt` par défaut) consommée par `train_ppo.py --load-value`.

### 4.9 `run_growing_pool.py`

Orchestrateur REINFORCE : découpe un entraînement en segments `train.py` successifs, chacun ajoutant son propre checkpoint au pool d'adversaires du segment suivant (avec PFSP). Ne contient aucune logique d'entraînement elle-même — seulement de la construction de ligne de commande et du pilotage de sous-processus.

- `_parse_final_winrates(log_path)` (`run_growing_pool.py:35`) : reparse la dernière ligne `"PFSP picks total: ..."` d'un log de segment (imprimée par `PFSPSampler.summary()`) pour récupérer le win-rate final de la policy contre chaque adversaire de ce segment. Appelée par `main()` uniquement si `--pool-max-size` est utilisé.
- `_prune_pool(pool_checkpoints, keep_every)` (`run_growing_pool.py:51`) : garde un checkpoint sur `keep_every` (par index de segment), plus toujours le plus récent — réduit la dilution du budget PFSP par adversaire à mesure que le pool grossit. Appelée par `main()` à chaque segment (`--pool-keep-every`).
- **`main()`** (`run_growing_pool.py:68`) : pour chaque segment, calcule le pool candidat (exclut `--pool-exclude`, applique `_prune_pool`, puis élague par faiblesse jusqu'à `--pool-max-size` si nécessaire en utilisant `_parse_final_winrates` du segment précédent — jamais le tout dernier ajouté), construit la commande `python -u train.py --load ... --opponent heuristic,<pool> --pfsp ...`, la lance via `subprocess.run(..., stdout=logf, stderr=subprocess.STDOUT)` en écrivant dans `seg_<N>.log`, s'arrête (`SystemExit`) si le segment échoue, sinon ajoute le checkpoint produit au pool et continue. `--start-segment` permet de reprendre une orchestration interrompue.

### 4.10 `run_growing_pool_ppo.py`

Équivalent PPO de 4.9 — même logique de segmentation/pool, pilote `train_ppo.py` au lieu de `train.py`. Plus simple : pas d'options d'élagage de pool (`--pool-keep-every`/`--pool-max-size`/`--pool-exclude` n'existent pas ici, seulement dans la version REINFORCE — c'est un premier essai, jamais encore testé avec ces raffinements).

- **`main()`** (`run_growing_pool_ppo.py:44`) : même structure que `run_growing_pool.py::main` mais sans les fonctions d'élagage — le pool grandit sans limite. Gère `--load-value` : le premier segment part de `--init-load` (policy seule, critic aléatoire) ; les segments suivants rechargent policy **et** critic depuis le même fichier de segment précédent (`--load` et `--load-value` pointent vers le même chemin, car `PPOPolicy.save()` écrit les deux dans un seul fichier) — sauf pour `--architecture shared`/`shared_aux`, où `--load-value` n'a pas de sens (un seul réseau) et n'est jamais passé.

### 4.11 `eval_policy.py`

Évalue un ou plusieurs checkpoints (mode glouton) contre `HeuristicPlayer` seul, sur exactement le même jeu de donnes pour une comparaison équitable.

- `_leads_to_all_pass(hands, dealer)` (`eval_policy.py:46`) : rejoue une donne donnée avec 4 `HeuristicPlayer` (`bid()` sans aléa, donc valable pour n'importe quel checkpoint qui hérite de cette même `bid()` via `RLPlayer`) et détecte si `GameEngine.run_auction()` a déclenché sa redonne interne (`engine.history['deal_hands'] != hands`, cf. `game.py:139-143`). Appelée par `generate_fixed_deals`.
- **`generate_fixed_deals(n_games, seed)`** (`eval_policy.py:58`) : génère `n_games` `(mains, dealer)` **une seule fois**, indépendamment de tout checkpoint, en écartant celles qui provoqueraient une redonne interne — indispensable pour que `deal(hands=...)` donne exactement les mêmes cartes à chaque checkpoint évalué. C'est la correction d'un piège réel documenté dans le module : sans mains fixées à l'avance, dès qu'un checkpoint joue une carte différente d'un autre, `HeuristicPlayer._choose_defausse_suit()` (seul point d'aléa en cours de jeu) consomme un nombre différent de tirages `random`, ce qui fait diverger tout le flux aléatoire global pour le reste des parties comparées. Appelée par `main()` et par `eval_matchup.py::main`.
- `run_fixed_episode(players_factory, hands, dealer)` (`eval_policy.py:78`) : joue une donne à mains fixées, retourne `team_points[0] - team_points[1]`. Appelée par `evaluate_fixed`.
- **`evaluate_fixed(players_factory, deals, seed)`** (`eval_policy.py:87`) : joue toutes les `deals` avec `players_factory`, reseed `random` avant (pour que l'aléa résiduel de la défausse heuristique reparte du même point pour chaque checkpoint comparé), retourne `(avg, win_rate)`. Appelée par `main()` et réutilisée directement par `eval_matchup.py`.
- **`main()`** (`eval_policy.py:102`) : `checkpoints` (liste positionnelle, `'heuristic'` valant comme pseudo-checkpoint de référence), `--architecture` (s'applique à tous les checkpoints non-`heuristic` de l'appel — mélanger plusieurs architectures nécessite plusieurs appels avec le même `--seed`/`--games`), génère les donnes fixées une fois (`generate_fixed_deals`), puis pour chaque checkpoint construit une `factory` (`HeuristicPlayer×4` ou `RLPlayer` autour d'une `PPOPolicy`/`SharedTrunkPPOPolicy`/`SharedTrunkAuxPPOPolicy` chargée en mode glouton) et appelle `evaluate_fixed`. Affiche un tableau trié par `avg` décroissant.

### 4.12 `eval_matchup.py`

Round-robin — chaque checkpoint contre **chaque autre** (pas seulement contre `heuristic`), nécessaire pour juger la polyvalence réelle d'un checkpoint entraîné en self-play (son `eval_avg` contre `heuristic` seul peut être trompeur — flat ou même en baisse — alors qu'il s'est en réalité renforcé contre d'autres styles de jeu, cf. `rl_experiments_v3/README.md`). Réimporte `generate_fixed_deals`/`evaluate_fixed` de `eval_policy.py` telles quelles.

- `parse_spec(raw, default_architecture)` (`eval_matchup.py:39`) : décode un argument CLI — `'heuristic'`, ou `architecture:chemin` (préfixe explicite, ex. `big:seg_300000.pt`), ou un chemin nu qui utilise `--architecture`. Permet de mélanger plusieurs architectures dans un **même** round-robin. Appelée par `main()`.
- `load_spec(path, architecture='small')` (`eval_matchup.py:52`) : `None` pour `'heuristic'`, sinon instancie la bonne classe de policy (`PPOPolicy`/`SharedTrunkPPOPolicy`/`SharedTrunkAuxPPOPolicy` selon l'architecture) et la charge en mode glouton — une seule fois par spec, réutilisée pour tous les appariements qui l'impliquent. Appelée par `main()`.
- `make_player(path, policy, name)` (`eval_matchup.py:72`) : `HeuristicPlayer` ou `RLPlayer(name, policy)` selon que `path == 'heuristic'`. Appelée par `matchup_factory`.
- `matchup_factory(path_a, policy_a, path_b, policy_b)` (`eval_matchup.py:78`) : construit la `factory` d'un appariement — A aux sièges 0/2, B aux sièges 1/3 (même convention que `eval_policy.py`). Appelée par `main()` pour chaque paire ordonnée `(a, b)` avec `a != b`.
- **`main()`** (`eval_matchup.py:90`) : parse les `specs`, génère les donnes fixées une fois, charge chaque policy une seule fois (`load_spec`), puis pour **chaque paire ordonnée** appelle `evaluate_fixed` (réimportée de `eval_policy.py`) via `matchup_factory`, et affiche à la fois le détail ligne par ligne et un tableau croisé récapitulatif. Note méthodologique implicite (documentée dans `CLAUDE.md`, pas dans ce fichier) : `skill(X,Y) = ((X vs Y) − (Y vs X)) / 2` corrige le biais de position de siège (~2-5 points en faveur des sièges 0/2) avant de comparer deux checkpoints — ce calcul n'est **pas** fait par le script lui-même, à faire manuellement sur sa sortie.

### 4.13 `play_test.py`

Le script le plus simple pour lancer une partie isolée sans interaction humaine et en garder une trace JSON — le premier réflexe de test après une modification de `game.py`/`players.py`.

- `load_hands(path)` (`play_test.py:8`) : charge et valide un fichier `--hands-file` (liste de 4 listes de chaînes carte). Appelée par `main()`.
- `build_players(strategies)` (`play_test.py:16`) : parse `--strategies` (4 valeurs séparées par des virgules), construit chaque joueur via `create_player` (players.py), et attache une `SimplePolicy()` **partagée** à tout `RLPlayer` sans policy — donc la stratégie `'rl'` ici est toujours une politique aléatoire non entraînée, jamais un vrai checkpoint (contrairement à `play_interactive.py`, qui sait charger `rl:architecture:chemin`). Appelée par `run_game`.
- `ta_points_to_normal_scale(raw_points)` (`play_test.py:29`) : convertit un score TA (échelle à 248/15 points) vers l'échelle normale à 162/10 points, pour un affichage comparable aux contrats couleur/SA — logique d'affichage uniquement, **sans effet sur le scoring réel** calculé par `GameEngine.play()`. Appelée par `main()` juste avant l'affichage final.
- **`run_game(strategies, hands, dealer, out_path)`** (`play_test.py:39`) : construit les joueurs, instancie un `CoincheEnv`, appelle `reset(hands)` puis `step(0)` (l'action passée est ignorée, cf. §4.3), écrit l'historique en JSON si `out_path` est fourni. Retourne `(contract, team_points, history)`. Appelée uniquement par `main()`.
- **`main()`** (`play_test.py:53`) : parse les arguments CLI, appelle `run_game`, affiche le contrat final et les points de l'équipe preneuse (converti via `ta_points_to_normal_scale` si TA).

### 4.14 `play_interactive.py`

Partie jouée au clavier contre des bots, avec conseil d'IA en direct et génération d'une relecture HTML.

- `load_hands(path)` (`play_interactive.py:12`) : identique à `play_test.load_hands` (dupliquée, pas importée — les deux scripts sont volontairement indépendants).
- **`_load_rl_policy(spec)`** (`play_interactive.py:20`) : parse `spec = 'architecture:chemin'` (ex. `'big:rl_experiments_v4/.../seg_750000.pt'`), instancie `NeuralPolicy(net_cls=CardNet ou CardNetBig)` et **charge réellement le checkpoint** en mode glouton (`record=False`) — c'est ici, contrairement à `play_test.py`, qu'un vrai checkpoint entraîné peut être joué. Appelée par `build_one_player`.
- `build_one_player(strat, name)` (`play_interactive.py:40`) : si `strat` commence par `'rl:'`, construit un `RLPlayer` autour de `_load_rl_policy(...)` ; sinon délègue à `create_player` (players.py), avec le même repli `SimplePolicy` que `play_test.py` pour un `RLPlayer` sans policy. Appelée par `build_players` et directement par `main()` pour construire l'`advisor`.
- `build_players(strategies)` (`play_interactive.py:49`) : parse `--strategies` (4 valeurs) et appelle `build_one_player` pour chacune. Appelée par `main()`.
- `_with_suffix(path, suffix)` (`play_interactive.py:54`) : insère un suffixe avant l'extension d'un chemin (ex. pour générer `game_interactive_botreplay.json` à côté de `game_interactive.json`). Appelée par `main()` pour le replay bot optionnel.
- **`main()`** (`play_interactive.py:59`) : détermine le siège humain (`human_idx`) et le spec de bot de référence (`bot_spec`, pour l'advisor par défaut et le replay), construit les joueurs, attache un `advisor` au joueur humain si `--advisor != 'none'`, construit un `GameEngine` et déroule `deal()` → `run_auction()` → `play()` directement (**pas** via `CoincheEnv`, contrairement à `play_test.py` — ce script a besoin d'accéder à `engine.dealer`/`engine.taker_idx`/`engine.contract` en détail pour l'affichage), sauvegarde le JSON, appelle `render_history.generate_page` sauf `--no-render`, puis propose interactivement de rejouer exactement la même donne + le même contrat avec des bots à la place de l'humain (`replay_engine`, en copiant `contract`/`taker_idx` sur un nouvel engine plutôt que de relancer une enchère qui pourrait diverger).

### 4.15 `render_history.py`

Transforme un historique JSON (`GameEngine.history`, produit par `play_test.py`, `play_interactive.py`, ou un script ad hoc) en une page HTML autonome avec un onglet par pli.

- `card_html(card)` (`render_history.py:18`) : une chaîne carte (`"AP"`) → un `<span>` coloré par couleur (`CARD_BG`). Appelée par `hand_html`/les boucles d'affichage de plis.
- `sort_hand(cards, trump)` (`render_history.py:27`) : trie une main pour l'affichage — par couleur (ordre fixe `P/C/K/T`) puis par force réelle dans chaque couleur (`TRUMP_ORDER` si atout/TA, `NORMAL_ORDER` sinon). Appelée par `player_box_html` et par la construction des mains initiales dans `generate_page`.
- `hand_html(cards)` (`render_history.py:43`) : joint une liste de `card_html(...)`, ou `"(vide)"`. Appelée par `player_box_html`.
- `card_point(card, trump)` (`render_history.py:49`) : réimplémentation autonome (pas un import de `GameEngine.card_point` — ce module ne dépend que des tables de constantes de `game.py`, pas de `GameEngine` lui-même) de la valeur en points d'une carte. Appelée par `running_team_scores`.
- **`running_team_scores(history, trump)`** (`render_history.py:61`) : recalcule, pli par pli, le score cumulé (points de cartes + belote + 10 de der), en répliquant **exactement** la logique de `GameEngine.play()` — y compris la détection de capot (tous les plis gagnés par la même équipe) et le siège belote (Roi+Dame d'atout tenus initialement par le même joueur, créditée dès que la 2ᵉ des deux cartes est jouée, quel que soit le vainqueur du pli). Retourne `(running: liste de (score0, score1) par pli, belote_info)`. Appelée par `generate_page`.
- **`coinche_final_scores(history, raw_final_scores, belote_info)`** (`render_history.py:115`) : si le contrat est coinché, remplace le score brut de plis par le forfait fixe (`160 + 2×contrat` pour le vainqueur, `0` pour l'autre) — réplique la branche coinche de `GameEngine.play()`. Retourne `None` si le contrat n'est pas coinché. Appelée par `generate_page`.
- `build_stages(history)` (`render_history.py:136`) : reconstruit, pli par pli, l'état des mains (en retirant les cartes jouées au fur et à mesure) et la liste des plis déjà complétés — la structure de données que chaque panneau HTML affiche. Appelée par `generate_page`.
- `player_box_html(seat, css_class, hand_cards, trump, leader_seat=None)` (`render_history.py:156`) : le bloc HTML d'un siège (nom, main triée, badge « entame » si ce siège a mené le pli). Appelée par `generate_page` (4 fois par panneau de pli).
- **`generate_page(history, out_path)`** (`render_history.py:164`) : la fonction principale — assemble `build_stages`, `running_team_scores`, `coinche_final_scores`, les blocs d'enchères et de mains initiales, un panneau HTML par pli (avec badges belote/entame), injecte le tout dans un template HTML/CSS/JS auto-contenu (onglets cliquables en JS vanilla, pas de dépendance externe) et écrit le résultat dans `out_path`. Appelée par `main()` **et directement par `play_interactive.py::main`** (import de `render_history.generate_page`) — c'est le seul point de couplage entre les deux scripts.
- **`main()`** (`render_history.py:342`) : `--history`/`--out`, charge le JSON et appelle `generate_page`. Point d'entrée de la commande `python3 render_history.py --history ... --out ...`.
