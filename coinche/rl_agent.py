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
    """4 suit "slots" symmetric with respect to trump, each with its own rank
    order (8 positions): for a suit contract, trump always occupies slot 0
    (TRUMP_ORDER), the 3 other suits follow in the fixed order of SUITS
    (NORMAL_ORDER); in TA all suits are trump (all 4 slots in TRUMP_ORDER);
    in SA none is (all 4 slots in NORMAL_ORDER). This way a given
    input/output position of the network always keeps the same strategic
    meaning ("best card of the trump slot", etc.) regardless of the donne's
    actual trump suit -- the network no longer has to relearn the same
    concept 4 times (once per physical suit)."""
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
    """Cards from the donne's already-completed tricks (cf.
    HeuristicPlayer._played_cards), read from the history ('RankSuit' ->
    canonical slot), as a 32-dim multi-hot vector."""
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
    """Number of cards present (0-8) per canonical slot, from a 32-dim
    multi-hot vector already split into 4 blocks of 8 by `_canonical_slots`."""
    return [sum(vec32[s * 8:(s + 1) * 8]) for s in range(4)]


def _record_void_from_trick(seat_suit_pairs, void, suits):
    """Marks as void in the suit led any seat that played a card of a
    different suit in this trick -- certain info (not a guess), since
    `legal_moves()` requires following the suit led whenever still possible."""
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
    """For each of the 3 other seats, in order relative to my own seat (next
    opponent, partner, previous opponent -- never myself, since my own suit
    composition is already known via hand_vec), and for each of the 4
    canonical slots: has this seat already revealed itself void in this
    suit, by not following it when it was led. Unlike the global count from
    `_slot_counts`/`unknown_vec`, this is exact info specific to one seat,
    that stays valid until the end of the donne."""
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
    """For each of the 3 other seats (order relative to my seat, as for
    voids): did they bid/raise this suit at least once during the auction
    (12 dims), and did they bid SA / TA at least once (6 dims) -- even if
    the final bid ended up elsewhere. Used to guess who holds trump or
    aces/jacks from what was bid, not just from the final retained contract."""
    engine = getattr(player, 'engine', None)
    seat = getattr(player, 'seat', 0)
    bid_suit = [[False] * 4 for _ in range(4)]
    bid_no_trump = [[False, False] for _ in range(4)]  # [seat] -> [bid SA, bid TA]

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
    """For each of the 3 other seats: have they already led (opened) a
    trick in this suit at least once this donne. Distinct from simply
    counting played cards (`played_vec`): captures a specific seat's suit
    preference/strength (heuristiques.md §2.1.2: replaying the suit where
    the partner made their first lead), info that `played_vec` alone can't
    reconstruct since it doesn't keep the "who led which trick" structure."""
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
    """Card points already banked by my side and by the opponents in the
    COMPLETE tricks of this donne (the current trick isn't won by anyone
    yet), normalized by a reference total (152 card points + 10 for the
    last trick = 162 -- the real scale differs slightly in TA/SA, but
    `trump_vec` lets the network contextualize). Includes neither the
    last-trick bonus (known only at the very end of the last trick) nor the
    belote (cf. the gap documented in CLAUDE.md between "credited to the
    trick" and the actual implementation on the initial hand, out of scope
    here) -- just the raw card points of tricks already won, to stay simple
    and unambiguous."""
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
    """Seat currently winning the trick in progress (before `player` plays
    their card), or None if the trick is empty. Reuses the engine's
    `card_order_key` rather than reimplementing a third card-strength logic."""
    if not trick or engine is None:
        return None
    lead_suit = trick[0][1].suit
    best = trick[0]
    for t in trick[1:]:
        if engine.card_order_key(t[1], lead_suit, trump) > engine.card_order_key(best[1], lead_suit, trump):
            best = t
    return best[0]


def encode_state(player, trick, trump, ablate_points=False) -> list:
    """Encodes the state seen by `player` when choosing a card: their hand,
    the cards already played in the donne, the current trick, the
    contract, and whether their side is attacking/took the contract.
    Always the same size (STATE_DIM), regardless of how many cards remain
    in hand or have already been played in the trick.

    The hand/played/trick blocks are no longer indexed by absolute
    physical suit but by canonical slot relative to trump
    (`_canonical_slots`): the symmetry between the 4 physical suits is
    thus provided by construction rather than left for the network to
    discover from the data.

    Derived features are added on top, summarizing what a real player
    naturally computes (suit length, remaining unknown cards, contract
    level, coinche, partner winning the trick, trick points already
    banked per side): the network no longer has to reconstruct them
    itself by counting from the raw multi-hot vectors, which speeds up
    learning without removing the underlying raw information."""
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

    # Suit length: number of cards in this slot at the initial deal (not
    # the current hand, which trivially shrinks over the donne and is
    # already captured in hand_vec) -> a stable property of the starting
    # hand, as for discarding on the HeuristicPlayer side (cf. self.initial_hand).
    initial_hand = getattr(player, 'initial_hand', player.hand)
    initial_vec = _relative_multi_hot(initial_hand, suits, orders)
    length_vec = [c / 8.0 for c in _slot_counts(initial_vec)]

    # Remaining unknown cards: neither in my current hand, nor already
    # played (current trick included) -> still distributed among partner and opponents.
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

    # Card points already banked by my side / the opponents in complete
    # tricks -- previously absent from the state (remarques_rl.md) even
    # though it's a direct signal for the critic (close to the quantity it
    # regresses). `ablate_points` (ablation study): forces this block to
    # zero without changing STATE_DIM, to isolate this feature's effect on
    # the actor all else being equal (same network, same protocol).
    points_vec = [0.0, 0.0] if ablate_points else _points_so_far(player, trump, engine)

    winner_seat = _current_trick_winner_seat(trick, trump, engine)
    partner_winning = 1.0 if (winner_seat is not None and winner_seat == (seat + 2) % 4) else 0.0

    # Voids: which seats (other than myself) have already revealed
    # themselves void in which suit, by not following it -- certain info,
    # unlike the partner/opponent ambiguity tolerated elsewhere for unseen cards.
    void_vec = _void_vec(player, trick, suits)

    # Bidding: who (among the 3 other seats) bid/raised which suit or
    # SA/TA -- signal of who holds trump or aces/jacks.
    auction_vec = _auction_signals_vec(player, suits)

    # Who has already led (opened) a trick in which suit -- suit
    # preference/strength per seat, distinct from simply counting played cards.
    led_vec = _led_suit_vec(player, trick, suits)

    return (hand_vec + played_vec + trick_vec + lead_vec + pos_vec + trump_vec
            + [is_attacker, is_taker] + length_vec + unknown_vec
            + [level_scalar, coinched_scalar, partner_winning] + points_vec + void_vec
            + auction_vec + led_vec)


STATE_DIM = 32 + 32 + 32 + 4 + 4 + 6 + 2 + 4 + 4 + 3 + 2 + 12 + 18 + 12  # 167


def _other_hands_vec(player, suits, orders) -> list:
    """CURRENT hands of the 3 other seats (order relative to my seat, as
    for void_vec/auction_vec/led_vec) -- privileged information the actor
    never sees (it only knows its own hand), reserved for the centralized
    critic (cf. encode_full_state): during training simulation all 4 hands
    are known to the process, even though a real player would never see them."""
    engine = getattr(player, 'engine', None)
    seat = getattr(player, 'seat', 0)
    v = []
    for offset in (1, 2, 3):
        other_seat = (seat + offset) % 4
        other_hand = engine.players[other_seat].hand if engine is not None else []
        v.extend(_relative_multi_hot(other_hand, suits, orders))
    return v


def _other_trump_counts(player, suits) -> list:
    """Number of cards currently in hand in the canonical trump slot (slot
    0, cf. _canonical_slots -- real trump for a suit contract; a
    degenerate but well-defined notion in SA/TA where all slots follow
    the same convention), for each of the 3 other seats -- same relative
    order as _other_hands_vec. Training target for
    SharedTrunkActorCriticAux's auxiliary task, never visible to the
    actor in real play (privileged info, only available in simulation)."""
    engine = getattr(player, 'engine', None)
    seat = getattr(player, 'seat', 0)
    trump_suit = suits[0]
    counts = []
    for offset in (1, 2, 3):
        other_seat = (seat + offset) % 4
        other_hand = engine.players[other_seat].hand if engine is not None else []
        counts.append(float(sum(1 for c in other_hand if c.suit == trump_suit)))
    return counts


def other_trump_counts(player, trump) -> list:
    """See _other_trump_counts -- computes suits/orders itself (same
    calling convention as encode_full_state)."""
    suits, _orders = _canonical_slots(trump)
    return _other_trump_counts(player, suits)


def encode_full_state(player, trick, trump, ablate_points=False) -> list:
    """CENTRALIZED state for the critic: encode_state() (what a real
    player sees) + the current hands of the 3 other seats -- privileged
    information, only available because we simulate the whole donne
    during training, never usable by the actor in real play. Analogous to
    a human player reviewing the complete donne afterward to understand
    what should have been done: the critic learns from this complete
    view, the actor keeps deciding from its partial view (encode_state)
    -- the two are independent networks (ValueNet vs CardNet,
    train_ppo.py), so nothing prevents this information asymmetry."""
    suits, orders = _canonical_slots(trump)
    return encode_state(player, trick, trump, ablate_points=ablate_points) + _other_hands_vec(player, suits, orders)


FULL_STATE_DIM = STATE_DIM + 3 * 32  # 167 + 96 = 263


class SimplePolicy:
    """Random policy (among legal moves): fallback without torch, and
    reference opponent to check that a trained policy is actually improving."""
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
            self.fc2 = nn.Linear(hidden, 32)  # one logit per canonical slot (relative suit x rank)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    class CardNetBig(nn.Module):
        """Deeper variant of CardNet (remarques_rl.md point 6): tests
        whether a single 128-unit hidden layer is a capacity ceiling
        (imitation saturates at ~92-93% accuracy, not 99%, with CardNet).
        Exists only alongside CardNet (never modified in place) so
        already-validated configs stay loadable/reproducible exactly as
        they were.

        Pre-LN LayerNorm (before each linear layer, not after): stabilizes
        training of a deeper MLP, same motivation as transformers' Pre-LN
        blocks."""
        def __init__(self, in_dim=STATE_DIM, hidden1=128, hidden2=128, hidden3=64):
            super().__init__()
            self.ln1 = nn.LayerNorm(in_dim)
            self.fc1 = nn.Linear(in_dim, hidden1)
            self.ln2 = nn.LayerNorm(hidden1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.ln3 = nn.LayerNorm(hidden2)
            self.fc3 = nn.Linear(hidden2, hidden3)
            self.ln4 = nn.LayerNorm(hidden3)
            self.fc4 = nn.Linear(hidden3, 32)

        def forward(self, x):
            x = F.gelu(self.fc1(self.ln1(x)))
            x = F.gelu(self.fc2(self.ln2(x)))
            x = F.gelu(self.fc3(self.ln3(x)))
            return self.fc4(self.ln4(x))

    class SharedTrunkActorCritic(nn.Module):
        """Step 2 of the plan (remarques_rl.md point 6): shared trunk
        between the actor and the centralized critic, so the privileged
        info (the 3 others' hands, cf. encode_full_state) can finally
        influence the actor's representation -- unlike the centralized
        critic already tested (ValueNet independent of CardNet, with no
        possible effect on the actor by construction, cf.
        rl_experiments_v3/README.md).

        Shared trunk (ln1/fc1/ln2/fc2, 167->128->128): same parameter
        names and shapes as CardNetBig's first 2 layers, so an
        already-trained imit_bignet*.pt can be reused as a starting point
        (cf. load_actor_from_cardnetbig below).

        Actor head (ln3_actor/fc3_actor/ln4_actor/fc4_actor, 128->64->32):
        identical to CardNetBig's second half -- same remapping possible
        from an existing CardNetBig checkpoint.

        Critic head: additionally receives the centralized info (the 3
        others' hands, 96-dim, already in encode_full_state) through a
        small, separate single-layer branch (ln_central/fc_central,
        96->32 -- deliberately shallow, cf. the 2026-07-27 discussion: a
        more structural than strategic role, the useful depth is already
        in the shared trunk and the critic head itself), concatenated to
        the shared trunk's output (128+32=160) before 1 private hidden
        layer (ln3_critic/fc3_critic, 160->64) then the scalar output
        (ln4_critic/fc4_critic, 64->1)."""
        def __init__(self, in_dim=STATE_DIM, other_dim=FULL_STATE_DIM - STATE_DIM,
                     trunk_hidden=128, actor_hidden=64, central_hidden=32, critic_hidden=64):
            super().__init__()
            # Shared trunk (identical in shape/names to CardNetBig's first 2 layers).
            self.ln1 = nn.LayerNorm(in_dim)
            self.fc1 = nn.Linear(in_dim, trunk_hidden)
            self.ln2 = nn.LayerNorm(trunk_hidden)
            self.fc2 = nn.Linear(trunk_hidden, trunk_hidden)

            # Actor head (identical in shape/names to CardNetBig's second half).
            self.ln3_actor = nn.LayerNorm(trunk_hidden)
            self.fc3_actor = nn.Linear(trunk_hidden, actor_hidden)
            self.ln4_actor = nn.LayerNorm(actor_hidden)
            self.fc4_actor = nn.Linear(actor_hidden, 32)

            # Centralized info branch (critic only) + critic head.
            self.ln_central = nn.LayerNorm(other_dim)
            self.fc_central = nn.Linear(other_dim, central_hidden)
            self.ln3_critic = nn.LayerNorm(trunk_hidden + central_hidden)
            self.fc3_critic = nn.Linear(trunk_hidden + central_hidden, critic_hidden)
            self.ln4_critic = nn.LayerNorm(critic_hidden)
            self.fc4_critic = nn.Linear(critic_hidden, 1)

            self._state_dim = in_dim

        def _trunk(self, x):
            x = F.gelu(self.fc1(self.ln1(x)))
            x = F.gelu(self.fc2(self.ln2(x)))
            return x

        def forward_actor(self, x):
            h = self._trunk(x)
            h = F.gelu(self.fc3_actor(self.ln3_actor(h)))
            return self.fc4_actor(self.ln4_actor(h))

        def forward(self, x):
            """Alias for forward_actor -- so NeuralPolicy (train.py, calls
            net(x) directly) can load this checkpoint as a frozen greedy
            opponent in a growing pool (cf. make_opponent_factory): only
            the actor head matters for an opponent, which is never updated."""
            return self.forward_actor(x)

        def forward_critic(self, x_full):
            x = x_full[..., :self._state_dim]
            other = x_full[..., self._state_dim:]
            h = self._trunk(x)
            c = F.gelu(self.fc_central(self.ln_central(other)))
            h = torch.cat([h, c], dim=-1)
            h = F.gelu(self.fc3_critic(self.ln3_critic(h)))
            return self.fc4_critic(self.ln4_critic(h)).squeeze(-1)

        def load_actor_from_cardnetbig(self, path, map_location='cpu'):
            """Loads the shared trunk + actor head from an already-trained
            CardNetBig checkpoint (e.g. imit_bignet_ent02.pt) -- same
            shape, simple name remapping (fc3/ln3/fc4/ln4 -> *_actor). The
            critic part (centralized branch + critic head) stays randomly
            initialized: nothing equivalent exists in a CardNetBig
            checkpoint (an independent network, never trained with the
            centralized info)."""
            obj = torch.load(path, map_location=map_location)
            state_dict = obj['policy'] if isinstance(obj, dict) and 'policy' in obj else obj
            remap = {
                'fc1.weight': 'fc1.weight', 'fc1.bias': 'fc1.bias',
                'ln1.weight': 'ln1.weight', 'ln1.bias': 'ln1.bias',
                'fc2.weight': 'fc2.weight', 'fc2.bias': 'fc2.bias',
                'ln2.weight': 'ln2.weight', 'ln2.bias': 'ln2.bias',
                'fc3.weight': 'fc3_actor.weight', 'fc3.bias': 'fc3_actor.bias',
                'ln3.weight': 'ln3_actor.weight', 'ln3.bias': 'ln3_actor.bias',
                'fc4.weight': 'fc4_actor.weight', 'fc4.bias': 'fc4_actor.bias',
                'ln4.weight': 'ln4_actor.weight', 'ln4.bias': 'ln4_actor.bias',
            }
            own_state = self.state_dict()
            for src_key, dst_key in remap.items():
                own_state[dst_key] = state_dict[src_key]
            self.load_state_dict(own_state)

    class SharedTrunkActorCriticAux(SharedTrunkActorCritic):
        """SharedTrunkActorCritic + a 3rd auxiliary head (rl_experiments_v5,
        2026-07-27 discussion): predicts the number of trumps remaining
        with each of the 3 other seats (cf. other_trump_counts), from the
        shared trunk ALONE (h, the output of _trunk) -- NOT from the
        critic's centralized info branch (fc_central/c, cf. forward_critic).

        This choice is deliberate: if the auxiliary head also saw c (as
        the critic head already does), the network could "cheat" --
        reading the answer almost directly out of c without needing to
        encode anything in h, which would empty the auxiliary task of its
        purpose (forcing the trunk SHARED with the actor to represent
        this info). Seeing only h, the only way to reduce this loss is to
        improve h itself -- its gradient therefore directly influences
        what the actor also uses, unlike the centralized critic (which
        has this shortcut via c, and so pushes more weakly on the trunk).

        A single hidden layer (same depth as the critic's centralized
        info branch, same reasoning: a more structural than strategic role)."""
        def __init__(self, in_dim=STATE_DIM, other_dim=FULL_STATE_DIM - STATE_DIM,
                     trunk_hidden=128, actor_hidden=64, central_hidden=32, critic_hidden=64,
                     aux_hidden=32, n_aux=3):
            super().__init__(in_dim=in_dim, other_dim=other_dim, trunk_hidden=trunk_hidden,
                              actor_hidden=actor_hidden, central_hidden=central_hidden,
                              critic_hidden=critic_hidden)
            self.ln_aux = nn.LayerNorm(trunk_hidden)
            self.fc_aux = nn.Linear(trunk_hidden, aux_hidden)
            self.ln_aux2 = nn.LayerNorm(aux_hidden)
            self.fc_aux_out = nn.Linear(aux_hidden, n_aux)

        def forward_aux(self, x):
            h = self._trunk(x)
            h = F.gelu(self.fc_aux(self.ln_aux(h)))
            return self.fc_aux_out(self.ln_aux2(h))

    class SharedTrunkActorCriticDeep(nn.Module):
        """Trunk/heads rebalancing of SharedTrunkActorCritic (the user's
        exact design, 2026-07-28 discussion): deeper trunk (3 layers
        instead of 2), shorter actor head (1 layer instead of 2).
        Independent class (not a subclass of SharedTrunkActorCritic --
        the trunk/critic-head structure is too different for a clean
        inheritance), same convention as CardNetBig/CardNet: never
        modifies existing classes in place.

        Shared trunk (ln1/fc1/ln2/fc2/ln3/fc3, 167->128->128->64): covers
        EXACTLY CardNetBig's first 3 layers (same names/shapes, unlike
        SharedTrunkActorCritic which only reused the first 2) -- same
        reasoning, being able to restart from an already-trained
        imit_bignet*.pt (cf. load_actor_from_cardnetbig).

        Actor head (ln4_actor/fc4_actor, 64->32): a single layer,
        CardNetBig's last one renamed -- so all the depth of CardNetBig's
        second half is handed to the shared trunk rather than the actor.

        Critic head: centralized info branch (ln_central/fc_central,
        96->64 -- now equal to the trunk's output dimension, a more
        balanced split than SharedTrunkActorCritic's 128 vs 32)
        concatenated to the trunk's output (64+64=128) before 1 private
        hidden layer (ln3_critic/fc3_critic, 128->32) then the scalar
        output (ln4_critic/fc4_critic, 32->1)."""
        def __init__(self, in_dim=STATE_DIM, other_dim=FULL_STATE_DIM - STATE_DIM,
                     trunk_hidden1=128, trunk_hidden2=128, trunk_hidden3=64,
                     central_hidden=64, critic_hidden=32):
            super().__init__()
            # Shared trunk (3 layers, identical in shape/names to CardNetBig's first 3).
            self.ln1 = nn.LayerNorm(in_dim)
            self.fc1 = nn.Linear(in_dim, trunk_hidden1)
            self.ln2 = nn.LayerNorm(trunk_hidden1)
            self.fc2 = nn.Linear(trunk_hidden1, trunk_hidden2)
            self.ln3 = nn.LayerNorm(trunk_hidden2)
            self.fc3 = nn.Linear(trunk_hidden2, trunk_hidden3)

            # Actor head (a single layer, CardNetBig's last one).
            self.ln4_actor = nn.LayerNorm(trunk_hidden3)
            self.fc4_actor = nn.Linear(trunk_hidden3, 32)

            # Centralized info branch (critic only) + critic head.
            self.ln_central = nn.LayerNorm(other_dim)
            self.fc_central = nn.Linear(other_dim, central_hidden)
            self.ln3_critic = nn.LayerNorm(trunk_hidden3 + central_hidden)
            self.fc3_critic = nn.Linear(trunk_hidden3 + central_hidden, critic_hidden)
            self.ln4_critic = nn.LayerNorm(critic_hidden)
            self.fc4_critic = nn.Linear(critic_hidden, 1)

            self._state_dim = in_dim

        def _trunk(self, x):
            x = F.gelu(self.fc1(self.ln1(x)))
            x = F.gelu(self.fc2(self.ln2(x)))
            x = F.gelu(self.fc3(self.ln3(x)))
            return x

        def forward_actor(self, x):
            h = self._trunk(x)
            return self.fc4_actor(self.ln4_actor(h))

        def forward(self, x):
            """Alias for forward_actor -- see SharedTrunkActorCritic.forward
            (same reason: loading as a frozen pool opponent via NeuralPolicy)."""
            return self.forward_actor(x)

        def forward_critic(self, x_full):
            x = x_full[..., :self._state_dim]
            other = x_full[..., self._state_dim:]
            h = self._trunk(x)
            c = F.gelu(self.fc_central(self.ln_central(other)))
            h = torch.cat([h, c], dim=-1)
            h = F.gelu(self.fc3_critic(self.ln3_critic(h)))
            return self.fc4_critic(self.ln4_critic(h)).squeeze(-1)

        def load_actor_from_cardnetbig(self, path, map_location='cpu'):
            """Loads the trunk (3 layers) + actor head (last layer) from
            an already-trained CardNetBig checkpoint -- here CardNetBig's
            4 layers cover EXACTLY trunk+head (167->128->128->64 then
            64->32), so a more direct remap than
            SharedTrunkActorCritic.load_actor_from_cardnetbig (only the
            last layer is renamed, fc1/ln1/fc2/ln2/fc3/ln3 already share
            the same names)."""
            obj = torch.load(path, map_location=map_location)
            state_dict = obj['policy'] if isinstance(obj, dict) and 'policy' in obj else obj
            own_state = self.state_dict()
            for key in ('fc1.weight', 'fc1.bias', 'ln1.weight', 'ln1.bias',
                        'fc2.weight', 'fc2.bias', 'ln2.weight', 'ln2.bias',
                        'fc3.weight', 'fc3.bias', 'ln3.weight', 'ln3.bias'):
                own_state[key] = state_dict[key]
            own_state['fc4_actor.weight'] = state_dict['fc4.weight']
            own_state['fc4_actor.bias'] = state_dict['fc4.bias']
            own_state['ln4_actor.weight'] = state_dict['ln4.weight']
            own_state['ln4_actor.bias'] = state_dict['ln4.bias']
            self.load_state_dict(own_state)

    class NeuralPolicy:
        """Policy trainable via REINFORCE. `record=True` (default) samples
        and stores the log-prob of each decision; `record=False`
        (evaluation) plays greedily (argmax) without touching the
        trajectory buffer."""
        def __init__(self, device='cpu', lr=1e-3, baseline_beta=0.95, entropy_beta=0.01, net_cls=CardNet):
            self.device = device
            self.net = net_cls().to(device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.record = True
            self.saved_log_probs = []
            self.saved_entropies = []
            self.baseline = 0.0
            self.baseline_beta = baseline_beta
            # Entropy bonus (cf. remarques_rl.md point 4): forces
            # exploration during REINFORCE even if the policy starts out
            # very confident after imitation pretraining run to convergence (pretrain.py).
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
            """One REINFORCE update per donne: loss = -(sum of the
            trajectory's log-probs, both team seats combined) * advantage,
            where the advantage is the reward minus a moving average
            (baseline) to reduce variance, minus an entropy bonus
            (`entropy_beta`) that pushes the policy to stay exploratory.
            Clears the trajectory buffers afterward."""
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
            """Accepts either a raw CardNet state_dict (imit.pt, REINFORCE
            segments) or a native PPO checkpoint (dict with a 'policy'
            key, cf. PPOPolicy.save() in train_ppo.py) -- needed so
            make_opponent_factory (train.py) can load a PPO checkpoint as
            a frozen opponent in a pool (run_growing_pool_ppo.py)."""
            obj = torch.load(path, map_location=self.device)
            self.net.load_state_dict(obj['policy'] if 'policy' in obj else obj)
