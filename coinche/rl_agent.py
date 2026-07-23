import random
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None

from .game import SUITS, TRUMP_ORDER, NORMAL_ORDER, Card

TRUMP_TYPES = ['P', 'C', 'K', 'T', 'SA', 'TA']


def _canonical_slots(trump):
    """4 « slots » de couleur symétriques par rapport à l'atout, chacun avec son
    ordre de rang (8 positions) : pour un contrat couleur, l'atout occupe toujours
    le slot 0 (ordre TRUMP_ORDER), les 3 autres couleurs suivent dans l'ordre fixe
    de SUITS (ordre NORMAL_ORDER) ; à TA toutes les couleurs sont atout (les 4
    slots en TRUMP_ORDER) ; à SA aucune ne l'est (les 4 slots en NORMAL_ORDER).
    Ainsi une même position d'entrée/sortie du réseau garde toujours le même sens
    stratégique (« meilleure carte du slot atout », etc.) quelle que soit la
    couleur d'atout réelle de la donne — le réseau n'a plus à réapprendre 4 fois
    (une par couleur physique) le même concept."""
    if trump in SUITS:
        suits = [trump] + [s for s in SUITS if s != trump]
        orders = [TRUMP_ORDER, NORMAL_ORDER, NORMAL_ORDER, NORMAL_ORDER]
    elif trump == 'TA':
        suits = list(SUITS)
        orders = [TRUMP_ORDER] * 4
    else:  # 'SA'
        suits = list(SUITS)
        orders = [NORMAL_ORDER] * 4
    return suits, orders


def _slot_index(suit, rank, suits, orders) -> int:
    slot = suits.index(suit)
    rank_idx = orders[slot].index(rank)
    return slot * 8 + rank_idx


def _relative_multi_hot(cards, suits, orders) -> list:
    v = [0.0] * 32
    for c in cards:
        v[_slot_index(c.suit, c.rank, suits, orders)] = 1.0
    return v


def _one_hot(index, size):
    v = [0.0] * size
    if index is not None and 0 <= index < size:
        v[index] = 1.0
    return v


def _played_before_this_trick(player, suits, orders) -> list:
    """Cartes des plis déjà complets de la donne (cf. HeuristicPlayer._played_cards),
    lues depuis l'historique ('RangSuit' -> slot canonique), sous forme multi-hot 32 dims."""
    engine = getattr(player, 'engine', None)
    v = [0.0] * 32
    if engine is None or not getattr(engine, 'history', None):
        return v
    for t in engine.history.get('tricks', []):
        for p in t['plays']:
            card_str = p['card']
            suit, rank = card_str[-1], card_str[:-1]
            v[_slot_index(suit, rank, suits, orders)] = 1.0
    return v


def _slot_counts(vec32) -> list:
    """Nombre de cartes présentes (0-8) par slot canonique, à partir d'un vecteur
    multi-hot 32 dims déjà découpé en 4 blocs de 8 par `_canonical_slots`."""
    return [sum(vec32[s * 8:(s + 1) * 8]) for s in range(4)]


def _record_void_from_trick(seat_suit_pairs, void, suits):
    """Marque comme "sec" (void) dans la couleur demandée tout siège qui a joué
    une carte d'une autre couleur dans ce pli -- info certaine (pas une
    supposition) puisque `legal_moves()` impose de fournir la couleur demandée
    dès qu'on le peut encore."""
    if not seat_suit_pairs:
        return
    lead_suit = seat_suit_pairs[0][1]
    if lead_suit not in suits:
        return
    lead_slot = suits.index(lead_suit)
    for seat, suit in seat_suit_pairs[1:]:
        if suit != lead_suit:
            void[seat][lead_slot] = True


def _void_vec(player, trick, suits) -> list:
    """Pour chacun des 3 autres sièges, dans l'ordre relatif à mon propre siège
    (adversaire suivant, partenaire, adversaire précédent -- jamais moi-même,
    puisque ma propre composition de couleurs est déjà connue via hand_vec), et
    pour chacun des 4 slots canoniques : ce siège s'est-il déjà révélé sec dans
    cette couleur, en ne la fournissant pas quand elle était demandée. Contrairement
    au comptage global de `_slot_counts`/`unknown_vec`, c'est une info exacte, propre
    à un siège précis, qui reste valable jusqu'à la fin de la donne."""
    engine = getattr(player, 'engine', None)
    seat = getattr(player, 'seat', 0)
    void = [[False] * 4 for _ in range(4)]

    if engine is not None and getattr(engine, 'history', None):
        for t in engine.history.get('tricks', []):
            pairs = [(p['seat'], p['card'][-1]) for p in t['plays']]
            _record_void_from_trick(pairs, void, suits)
    live_pairs = [(s, c.suit) for s, c in trick]
    _record_void_from_trick(live_pairs, void, suits)

    v = []
    for offset in (1, 2, 3):
        other_seat = (seat + offset) % 4
        v.extend(1.0 if void[other_seat][slot] else 0.0 for slot in range(4))
    return v


def _auction_signals_vec(player, suits) -> list:
    """Pour chacun des 3 autres sièges (ordre relatif à mon siège, comme pour les
    renonces) : a-t-il annoncé/remonté cette couleur au moins une fois pendant
    les enchères (12 dims), et a-t-il annoncé SA / TA au moins une fois (6 dims)
    -- même si l'enchère finale a fini ailleurs. Sert à deviner qui détient de
    l'atout ou des as/valets à partir de ce qui a été annoncé, pas seulement du
    contrat final retenu."""
    engine = getattr(player, 'engine', None)
    seat = getattr(player, 'seat', 0)
    bid_suit = [[False] * 4 for _ in range(4)]
    bid_no_trump = [[False, False] for _ in range(4)]  # [seat] -> [a annonce SA, a annonce TA]

    if engine is not None and getattr(engine, 'history', None):
        for entry in engine.history.get('auction', []):
            offer = entry.get('offer')
            if offer is None:
                continue
            bidder, trump_bid = entry['seat'], offer[1]
            if trump_bid in suits:
                bid_suit[bidder][suits.index(trump_bid)] = True
            elif trump_bid == 'SA':
                bid_no_trump[bidder][0] = True
            elif trump_bid == 'TA':
                bid_no_trump[bidder][1] = True

    v = []
    for offset in (1, 2, 3):
        other_seat = (seat + offset) % 4
        v.extend(1.0 if bid_suit[other_seat][slot] else 0.0 for slot in range(4))
    for offset in (1, 2, 3):
        other_seat = (seat + offset) % 4
        v.extend(1.0 if x else 0.0 for x in bid_no_trump[other_seat])
    return v


def _led_suit_vec(player, trick, suits) -> list:
    """Pour chacun des 3 autres sièges : a-t-il déjà mené (ouvert) un pli dans
    cette couleur au moins une fois cette donne. Distinct du simple comptage de
    cartes tombées (`played_vec`) : capture une préférence/force de couleur d'un
    siège précis (heuristiques.md §2.1.2 : rejouer la couleur où le partenaire a
    fait sa première ouverture), une info que `played_vec` seul ne permet pas de
    reconstituer puisqu'il ne garde pas la structure "qui a mené quel pli"."""
    engine = getattr(player, 'engine', None)
    seat = getattr(player, 'seat', 0)
    led = [[False] * 4 for _ in range(4)]

    def mark(leader_seat, led_suit):
        if led_suit in suits:
            led[leader_seat][suits.index(led_suit)] = True

    if engine is not None and getattr(engine, 'history', None):
        for t in engine.history.get('tricks', []):
            plays = t['plays']
            if plays:
                mark(plays[0]['seat'], plays[0]['card'][-1])
    if trick:
        mark(trick[0][0], trick[0][1].suit)

    v = []
    for offset in (1, 2, 3):
        other_seat = (seat + offset) % 4
        v.extend(1.0 if led[other_seat][slot] else 0.0 for slot in range(4))
    return v


def _points_so_far(player, trump, engine) -> list:
    """Points de cartes deja engranges par mon camp et par l'adversaire dans
    les plis COMPLETS de cette donne (le pli en cours n'est pas encore gagne
    par personne), normalises par un total de reference (152 points de cartes
    + 10 de der = 162 -- l'echelle reelle differe legerement a TA/SA, mais
    `trump_vec` permet au reseau de contextualiser). N'inclut ni le 10 de der
    (connu seulement a la toute fin du dernier pli) ni la belote (cf. l'ecart
    documente dans CLAUDE.md entre "credite au pli" et l'implementation
    reelle sur la main initiale, hors-scope ici) -- seulement les points de
    cartes bruts des plis deja remportes, pour rester simple et sans ambiguite."""
    seat = getattr(player, 'seat', 0)
    mine, opp = 0, 0
    if engine is not None and getattr(engine, 'history', None):
        for t in engine.history.get('tricks', []):
            trick_points = sum(
                engine.card_point(Card(p['card'][-1], p['card'][:-1]), trump)
                for p in t['plays']
            )
            if t['winner'] % 2 == seat % 2:
                mine += trick_points
            else:
                opp += trick_points
    return [mine / 162.0, opp / 162.0]


def _current_trick_winner_seat(trick, trump, engine):
    """Siège actuellement maître du pli en cours (avant que `player` ne joue sa
    carte), ou None si le pli est vide. Réutilise `card_order_key` du moteur
    plutôt que de réimplémenter une troisième logique de force de carte."""
    if not trick or engine is None:
        return None
    lead_suit = trick[0][1].suit
    best = trick[0]
    for t in trick[1:]:
        if engine.card_order_key(t[1], lead_suit, trump) > engine.card_order_key(best[1], lead_suit, trump):
            best = t
    return best[0]


def encode_state(player, trick, trump, ablate_points=False) -> list:
    """Encode l'état vu par `player` au moment de choisir une carte : sa main, les
    cartes déjà tombées dans la donne, le pli en cours, le contrat, et si son
    camp attaque/a pris le contrat. Toujours la même taille (STATE_DIM), quel
    que soit le nombre de cartes restantes en main ou déjà jouées dans le pli.

    Les blocs main/défausses/pli ne sont plus indexés par couleur physique
    absolue mais par slot canonique relatif à l'atout (`_canonical_slots`) : la
    symétrie entre les 4 couleurs physiques est ainsi apportée par construction
    plutôt que laissée à découvrir par le réseau depuis les données.

    S'y ajoutent des features dérivées résumant ce qu'un joueur réel calcule
    naturellement (longueur de couleur, cartes inconnues restantes, niveau du
    contrat, coinche, partenaire maître du pli, points de plis déjà engrangés
    par camp) : le réseau n'a plus à les reconstituer lui-même par comptage
    depuis les multi-hot bruts, ce qui accélère l'apprentissage sans retirer
    l'information brute sous-jacente."""
    engine = getattr(player, 'engine', None)
    taker = getattr(engine, 'taker_idx', None)
    seat = getattr(player, 'seat', 0)
    is_attacker = 1.0 if (taker is not None and taker % 2 == seat % 2) else 0.0
    is_taker = 1.0 if (taker == seat) else 0.0

    suits, orders = _canonical_slots(trump)
    hand_vec = _relative_multi_hot(player.hand, suits, orders)
    played_vec = _played_before_this_trick(player, suits, orders)
    trick_vec = _relative_multi_hot([c for _, c in trick], suits, orders)
    lead_slot = suits.index(trick[0][1].suit) if trick else None
    lead_vec = _one_hot(lead_slot, 4)
    pos_vec = _one_hot(len(trick), 4)
    trump_vec = _one_hot(TRUMP_TYPES.index(trump) if trump in TRUMP_TYPES else None, 6)

    # Longueur de couleur : nombre de cartes de ce slot à la donne initiale (pas
    # la main courante, qui décroît trivialement au fil de la donne et se
    # retrouve déjà dans hand_vec) -> une propriété stable du jeu de départ,
    # comme pour la défausse côté HeuristicPlayer (cf. self.initial_hand).
    initial_hand = getattr(player, 'initial_hand', player.hand)
    initial_vec = _relative_multi_hot(initial_hand, suits, orders)
    length_vec = [c / 8.0 for c in _slot_counts(initial_vec)]

    # Cartes inconnues restantes : ni dans ma main courante, ni déjà tombées
    # (pli en cours inclus) -> encore réparties entre partenaire et adversaires.
    hand_counts = _slot_counts(hand_vec)
    played_counts = _slot_counts(played_vec)
    trick_counts = _slot_counts(trick_vec)
    unknown_vec = [
        (8 - hand_counts[s] - played_counts[s] - trick_counts[s]) / 8.0
        for s in range(4)
    ]

    contract = getattr(engine, 'contract', None)
    level_scalar = (getattr(contract, 'level', 0) or 0) / 250.0
    coinched_scalar = 1.0 if getattr(contract, 'coinched', False) else 0.0

    # Points de cartes deja engranges par mon camp / l'adversaire dans les plis
    # complets -- absent jusqu'ici de l'etat (remarques_rl.md) alors que c'est
    # un signal direct pour le critic (approche la grandeur qu'il regresse).
    # `ablate_points` (etude d'ablation) : force ce bloc a zero sans changer
    # STATE_DIM, pour isoler l'effet de cette feature sur l'acteur toutes
    # choses egales par ailleurs (meme reseau, meme protocole).
    points_vec = [0.0, 0.0] if ablate_points else _points_so_far(player, trump, engine)

    winner_seat = _current_trick_winner_seat(trick, trump, engine)
    partner_winning = 1.0 if (winner_seat is not None and winner_seat == (seat + 2) % 4) else 0.0

    # Renonces : quels sièges (hors moi-même) se sont déjà révélés secs dans
    # quelle couleur, en ne la fournissant pas -- info certaine, contrairement à
    # l'ambiguïté partenaire/adversaire tolérée ailleurs sur les cartes non vues.
    void_vec = _void_vec(player, trick, suits)

    # Enchères : qui (parmi les 3 autres sièges) a annoncé/remonté quelle
    # couleur ou SA/TA -- signal de qui détient l'atout ou des as/valets.
    auction_vec = _auction_signals_vec(player, suits)

    # Qui a déjà mené (ouvert) un pli dans quelle couleur -- préférence/force de
    # couleur par siège, distincte du simple comptage de cartes tombées.
    led_vec = _led_suit_vec(player, trick, suits)

    return (hand_vec + played_vec + trick_vec + lead_vec + pos_vec + trump_vec
            + [is_attacker, is_taker] + length_vec + unknown_vec
            + [level_scalar, coinched_scalar, partner_winning] + points_vec + void_vec
            + auction_vec + led_vec)


STATE_DIM = 32 + 32 + 32 + 4 + 4 + 6 + 2 + 4 + 4 + 3 + 2 + 12 + 18 + 12  # 167


def _other_hands_vec(player, suits, orders) -> list:
    """Mains ACTUELLES des 3 autres sieges (ordre relatif a mon siege, comme
    void_vec/auction_vec/led_vec) -- information privilegiee que l'acteur ne
    voit jamais (il ne connait que sa propre main), reservee au critic
    centralise (cf. encode_full_state) : en simulation d'entrainement les 4
    mains sont connues du process, meme si un joueur reel ne les verrait pas."""
    engine = getattr(player, 'engine', None)
    seat = getattr(player, 'seat', 0)
    v = []
    for offset in (1, 2, 3):
        other_seat = (seat + offset) % 4
        other_hand = engine.players[other_seat].hand if engine is not None else []
        v.extend(_relative_multi_hot(other_hand, suits, orders))
    return v


def encode_full_state(player, trick, trump, ablate_points=False) -> list:
    """Etat CENTRALISE pour le critic : encode_state() (ce qu'un joueur reel
    voit) + les mains actuelles des 3 autres sieges -- information
    privilegiee, disponible seulement parce qu'on simule la donne en entier
    a l'entrainement, jamais utilisable par l'acteur en jeu reel. Analogue a
    un joueur humain qui revoit la donne complete apres coup pour comprendre
    ce qu'il aurait fallu faire : le critic apprend depuis cette vue
    complete, l'acteur continue de decider depuis sa vue partielle
    (encode_state) -- les deux sont des reseaux independants (ValueNet vs
    CardNet, train_ppo.py), donc rien n'empeche cette asymetrie d'info."""
    suits, orders = _canonical_slots(trump)
    return encode_state(player, trick, trump, ablate_points=ablate_points) + _other_hands_vec(player, suits, orders)


FULL_STATE_DIM = STATE_DIM + 3 * 32  # 167 + 96 = 263


class SimplePolicy:
    """Politique aléatoire (parmi les coups légaux) : repli sans torch, et
    adversaire de référence pour vérifier qu'une policy entraînée progresse."""
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.record = False

    def choose_card(self, player, legal, leader, trick, trump):
        return self.rng.choice(legal)


if torch is not None:
    class CardNet(nn.Module):
        def __init__(self, in_dim=STATE_DIM, hidden=128):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, 32)  # un logit par slot canonique (couleur relative x rang)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    class NeuralPolicy:
        """Policy entraînable par REINFORCE. `record=True` (par défaut) échantillonne
        et mémorise le log-prob de chaque décision ; `record=False` (évaluation)
        joue en glouton (argmax) sans toucher au buffer de trajectoire."""
        def __init__(self, device='cpu', lr=1e-3, baseline_beta=0.95, entropy_beta=0.01):
            self.device = device
            self.net = CardNet().to(device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.record = True
            self.saved_log_probs = []
            self.saved_entropies = []
            self.baseline = 0.0
            self.baseline_beta = baseline_beta
            # Bonus d'entropie (cf. remarques_rl.md point 4) : force l'exploration
            # pendant le REINFORCE meme si la policy demarre tres confiante apres un
            # pre-entrainement par imitation pousse a convergence (pretrain.py).
            self.entropy_beta = entropy_beta

        def choose_card(self, player, legal, leader, trick, trump):
            x = torch.tensor(encode_state(player, trick, trump), dtype=torch.float32, device=self.device)
            logits = self.net(x)
            suits, orders = _canonical_slots(trump)
            mask = torch.full((32,), float('-inf'), device=self.device)
            legal_idx = [_slot_index(c.suit, c.rank, suits, orders) for c in legal]
            mask[legal_idx] = 0.0
            masked_logits = logits + mask

            if self.record:
                dist = torch.distributions.Categorical(logits=masked_logits)
                action_idx = dist.sample()
                self.saved_log_probs.append(dist.log_prob(action_idx))
                self.saved_entropies.append(dist.entropy())
            else:
                action_idx = torch.argmax(masked_logits)

            slot, rank_pos = divmod(int(action_idx.item()), 8)
            chosen_suit, chosen_rank = suits[slot], orders[slot][rank_pos]
            return next(c for c in legal if c.suit == chosen_suit and c.rank == chosen_rank)

        def update(self, reward: float):
            """Une mise à jour REINFORCE par donne : loss = -(somme des log-probs de
            la trajectoire, les deux sièges de l'équipe confondus) * avantage, où
            l'avantage est la reward moins une moyenne mobile (baseline) pour
            réduire la variance, moins un bonus d'entropie (`entropy_beta`) qui
            pousse la policy à rester exploratoire. Vide les buffers de
            trajectoire après coup."""
            if not self.saved_log_probs:
                return
            advantage = reward - self.baseline
            entropy_bonus = torch.stack(self.saved_entropies).sum()
            loss = -torch.stack(self.saved_log_probs).sum() * advantage - self.entropy_beta * entropy_bonus
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.baseline = self.baseline_beta * self.baseline + (1 - self.baseline_beta) * reward
            self.saved_log_probs = []
            self.saved_entropies = []

        def save(self, path):
            torch.save(self.net.state_dict(), path)

        def load(self, path):
            self.net.load_state_dict(torch.load(path, map_location=self.device))
