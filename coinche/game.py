from typing import List, Tuple, Optional
import random

SUITS = ["P", "C", "K", "T"]
RANKS = ["A", "10", "K", "Q", "J", "9", "8", "7"]

TRUMP_ORDER = ["J", "9", "A", "10", "K", "Q", "8", "7"]
NORMAL_ORDER = ["A", "10", "K", "Q", "J", "9", "8", "7"]

TRUMP_POINTS = {"J":20, "9":14, "A":11, "10":10, "K":4, "Q":3, "8":0, "7":0}
NORMAL_POINTS = {"A":11, "10":10, "K":4, "Q":3, "J":2, "9":0, "8":0, "7":0}
SA_POINTS = {"A":19, "10":10, "K":4, "Q":3, "J":2, "9":0, "8":0, "7":0}

class Card:
    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return f"{self.rank}{self.suit}"

class Deck:
    def __init__(self):
        self.cards = [Card(s, r) for s in SUITS for r in RANKS]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self) -> List[List[Card]]:
        self.shuffle()
        hands = [self.cards[i*8:(i+1)*8] for i in range(4)]
        return hands

class Contract:
    def __init__(self, level:int=80, trump:Optional[str]='P', coinched:bool=False, capot:bool=False):
        self.level = level
        self.trump = trump
        self.coinched = coinched
        self.capot = capot

class GameEngine:
    def __init__(self, players, dealer: int = 0):
        self.players = players
        self.dealer = dealer
        self.hands = []
        self.tricks = []
        self.contract = None
        self.taker_idx = None
        self.history = None

    def deal(self, hands: Optional[List[List[str]]] = None):
        """Deal cards. If `hands` is provided, it should be list of 4 lists of strings like ['AP','10C',...]."""
        if hands is None:
            deck = Deck()
            self.hands = deck.deal()
        else:
            parsed = []
            for h in hands:
                hand_cards = []
                for s in h:
                    if len(s) >= 2:
                        suit = s[-1]
                        rank = s[:-1]
                        hand_cards.append(Card(suit, rank))
                parsed.append(hand_cards)
            while len(parsed) < 4:
                parsed.append([])
            self.hands = parsed

        for p, hand in zip(self.players, self.hands):
            p.deal(hand)
        for i,p in enumerate(self.players):
            setattr(p, 'seat', i)
            setattr(p, 'engine', self)

        # minimal history: deal_hands, auction, tricks
        self.history = {
            'deal_hands': [[repr(c) for c in h] for h in self.hands],
            'auction': [],
            'tricks': [],
        }

    def run_auction(self):
        passes = 0
        best = None
        idx = (self.dealer + 1) % 4
        while passes < 4:
            player = self.players[idx]
            b = player.bid(best)
            # record bid (None for pass) in history
            if self.history is not None:
                self.history['auction'].append({'seat': idx, 'offer': b})
            if b is None:
                passes += 1
            else:
                best = (idx, b)
                passes = 0
            idx = (idx+1)%4

        if best is None:
            # All players passed: redeal
            self.dealer = (self.dealer + 1) % 4
            self.deal()
            self.run_auction()
        else:
            bidder_idx, (level, trump, coinched, capot) = best
            self.contract = Contract(level, trump, coinched, capot)
            self.taker_idx = bidder_idx

        if self.history is not None:
            self.history['contract'] = {'level': self.contract.level, 'trump': self.contract.trump, 'taker': getattr(self, 'taker_idx', None)}

    def card_order_key(self, card:Card, lead_suit:Optional[str], trump:str):
        if trump == 'TA' or (card.suit == trump):
            order = TRUMP_ORDER
        elif trump == 'SA':
            order = NORMAL_ORDER
        else:
            order = NORMAL_ORDER
        try:
            rank_idx = order.index(card.rank)
        except ValueError:
            rank_idx = 99
        is_trump = 1 if (trump=='TA' or card.suit==trump) else 0
        is_lead = 1 if (lead_suit is not None and card.suit==lead_suit) else 0
        return (is_trump, is_lead, -rank_idx)

    def evaluate_trick(self, trick: List[Tuple[int, Card]], trump: str) -> int:
        lead_suit = trick[0][1].suit
        best = trick[0]
        for t in trick[1:]:
            if self.card_order_key(t[1], lead_suit, trump) > self.card_order_key(best[1], lead_suit, trump):
                best = t
        return best[0]

    def card_point(self, card:Card, trump:str):
        if trump == 'TA' or card.suit == trump:
            return TRUMP_POINTS.get(card.rank, 0)
        elif trump == 'SA':
            return SA_POINTS.get(card.rank, 0)
        else:
            return NORMAL_POINTS.get(card.rank, 0)

    def play(self):
        if self.contract is None:
            self.run_auction()
        trump = self.contract.trump
        leader = (self.dealer + 1) % 4
        tricks_won = {i: [] for i in range(4)}
        for _ in range(8):
            trick = []
            trick_record = {'plays': []}
            for i in range(4):
                idx = (leader + i) % 4
                player = self.players[idx]
                card = player.play_card(idx, leader, trick, trump)
                trick.append((idx, card))
                if self.history is not None:
                    trick_record['plays'].append({'seat': idx, 'card': repr(card)})
            winner = self.evaluate_trick(trick, trump)
            if self.history is not None:
                trick_record['winner'] = winner
                self.history['tricks'].append(trick_record)
            tricks_won[winner].extend([c for (_,c) in trick])
            leader = winner

        # scoring (unchanged)
        team_points = {0:0,1:0}
        for p, cards in tricks_won.items():
            pts = sum(self.card_point(c, trump) for c in cards)
            team = p%2
            team_points[team] += pts
        if trump != 'TA':
            last_winner = leader
            team_points[last_winner%2] += 10

        belote_bonus = {0:0,1:0}
        for p, cards in tricks_won.items():
            ranks = {(c.suit,c.rank) for c in cards}
            if (trump!='SA' and trump!='TA'):
                if (trump,'K') in ranks and (trump,'Q') in ranks:
                    belote_bonus[p%2] += 20
        for team in [0,1]:
            team_points[team] += belote_bonus[team]

        return team_points, self.contract