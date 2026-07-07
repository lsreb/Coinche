import random
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None

from .game import SUITS, RANKS

# Table canonique unique des 32 cartes : sert à la fois à encoder la main/les
# cartes déjà jouées, à définir l'espace d'actions du réseau (32 sorties) et à
# décoder un indice choisi vers une carte réelle de la main. Ne jamais dupliquer
# cet ordre ailleurs (un décalage entre encodage/sortie/décodage serait silencieux).
ALL_CARDS = [(s, r) for s in SUITS for r in RANKS]
CARD_TO_INDEX = {sr: i for i, sr in enumerate(ALL_CARDS)}
TRUMP_TYPES = ['P', 'C', 'K', 'T', 'SA', 'TA']


def _card_index(card) -> int:
    return CARD_TO_INDEX[(card.suit, card.rank)]


def _one_hot(index, size):
    v = [0.0] * size
    if index is not None and 0 <= index < size:
        v[index] = 1.0
    return v


def _multi_hot_cards(cards) -> list:
    v = [0.0] * 32
    for c in cards:
        v[_card_index(c)] = 1.0
    return v


def _played_before_this_trick(player) -> list:
    """Cartes des plis déjà complets de la donne (cf. HeuristicPlayer._played_cards),
    sous forme de reprs 'RangSuit' dans l'historique -> multi-hot 32 dims."""
    engine = getattr(player, 'engine', None)
    v = [0.0] * 32
    if engine is None or not getattr(engine, 'history', None):
        return v
    for t in engine.history.get('tricks', []):
        for p in t['plays']:
            card_str = p['card']
            suit, rank = card_str[-1], card_str[:-1]
            idx = CARD_TO_INDEX.get((suit, rank))
            if idx is not None:
                v[idx] = 1.0
    return v


def encode_state(player, trick, trump) -> list:
    """Encode l'état vu par `player` au moment de choisir une carte : sa main, les
    cartes déjà tombées dans la donne, le pli en cours, le contrat, et si son
    camp attaque/a pris le contrat. Toujours la même taille (112), quel que soit
    le nombre de cartes restantes en main ou déjà jouées dans le pli."""
    engine = getattr(player, 'engine', None)
    taker = getattr(engine, 'taker_idx', None)
    seat = getattr(player, 'seat', 0)
    is_attacker = 1.0 if (taker is not None and taker % 2 == seat % 2) else 0.0
    is_taker = 1.0 if (taker == seat) else 0.0

    hand_vec = _multi_hot_cards(player.hand)
    played_vec = _played_before_this_trick(player)
    trick_vec = _multi_hot_cards([c for _, c in trick])
    lead_suit_idx = SUITS.index(trick[0][1].suit) if trick else None
    lead_vec = _one_hot(lead_suit_idx, 4)
    pos_vec = _one_hot(len(trick), 4)
    trump_vec = _one_hot(TRUMP_TYPES.index(trump) if trump in TRUMP_TYPES else None, 6)

    return hand_vec + played_vec + trick_vec + lead_vec + pos_vec + trump_vec + [is_attacker, is_taker]


STATE_DIM = 32 + 32 + 32 + 4 + 4 + 6 + 2  # 112


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
            self.fc2 = nn.Linear(hidden, 32)  # un logit par carte possible du jeu

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    class NeuralPolicy:
        """Policy entraînable par REINFORCE. `record=True` (par défaut) échantillonne
        et mémorise le log-prob de chaque décision ; `record=False` (évaluation)
        joue en glouton (argmax) sans toucher au buffer de trajectoire."""
        def __init__(self, device='cpu', lr=1e-3, baseline_beta=0.95):
            self.device = device
            self.net = CardNet().to(device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.record = True
            self.saved_log_probs = []
            self.baseline = 0.0
            self.baseline_beta = baseline_beta

        def choose_card(self, player, legal, leader, trick, trump):
            x = torch.tensor(encode_state(player, trick, trump), dtype=torch.float32, device=self.device)
            logits = self.net(x)
            mask = torch.full((32,), float('-inf'), device=self.device)
            legal_idx = [_card_index(c) for c in legal]
            mask[legal_idx] = 0.0
            masked_logits = logits + mask

            if self.record:
                dist = torch.distributions.Categorical(logits=masked_logits)
                action_idx = dist.sample()
                self.saved_log_probs.append(dist.log_prob(action_idx))
            else:
                action_idx = torch.argmax(masked_logits)

            chosen_suit, chosen_rank = ALL_CARDS[int(action_idx.item())]
            return next(c for c in legal if c.suit == chosen_suit and c.rank == chosen_rank)

        def update(self, reward: float):
            """Une mise à jour REINFORCE par donne : loss = -(somme des log-probs de
            la trajectoire, les deux sièges de l'équipe confondus) * avantage, où
            l'avantage est la reward moins une moyenne mobile (baseline) pour
            réduire la variance. Vide le buffer de trajectoire après coup."""
            if not self.saved_log_probs:
                return
            advantage = reward - self.baseline
            loss = -torch.stack(self.saved_log_probs).sum() * advantage
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.baseline = self.baseline_beta * self.baseline + (1 - self.baseline_beta) * reward
            self.saved_log_probs = []

        def save(self, path):
            torch.save(self.net.state_dict(), path)

        def load(self, path):
            self.net.load_state_dict(torch.load(path, map_location=self.device))
