import random
from typing import List, Optional, Tuple
from .game import Card, SUITS, TRUMP_ORDER, NORMAL_ORDER


RANK_ORDER = {r: i for i, r in enumerate(NORMAL_ORDER)}

def count_suit(hand: List[Card], suit: str) -> int:
    return sum(1 for c in hand if c.suit == suit)

def has_rank(hand: List[Card], suit: str, rank: str) -> bool:
    return any(c.suit == suit and c.rank == rank for c in hand)

class Player:
    def __init__(self, name: str):
        self.name = name
        self.hand: List[Card] = []

    def deal(self, hand: List[Card]):
        self.hand = hand

    def bid(self, current_best: Optional[Tuple[int,str,bool,bool]]):
        # return (level:int, trump:str, coinched:bool, capot:bool) or None for pass
        return None

    def play_card(self, seat:int, leader:int, trick:List[Tuple[int,Card]], trump:str) -> Card:
        raise NotImplementedError()

class RandomPlayer(Player):
    def bid(self, current_best):
        # random pass or small bid
        if random.random() < 0.7:
            return None
        level = random.choice([80,90,100])
        trump = random.choice(["P","C","K","T","SA","TA"])
        return (level, trump, False, False)

    def play_card(self, seat:int, leader:int, trick:list, trump:str):
        # choose a legal card simple: follow suit if possible
        if not trick:
            choice = random.choice(self.hand)
        else:
            lead = trick[0][1].suit
            same = [c for c in self.hand if c.suit==lead]
            if same:
                choice = random.choice(same)
            else:
                # if no lead suit, try trump
                tr = [c for c in self.hand if c.suit==trump]
                if tr:
                    choice = random.choice(tr)
                else:
                    choice = random.choice(self.hand)
        self.hand.remove(choice)
        return choice

class HeuristicPlayer(Player):
    def __init__(self, name:str, variant: str = 'user'):
        super().__init__(name)
        self.variant = variant

    def bid(self, current_best):
        # Implement heuristics from heuristiques.md (simplified)
        # Try color bids first
        best_offer = None
        for s in SUITS:
            cnt = count_suit(self.hand, s)
            has_j = has_rank(self.hand, s, 'J')
            has_9 = has_rank(self.hand, s, '9')
            if cnt >= 4 and has_j and has_9:
                best_offer = (100, s, False, False)
                break
            if cnt >= 3 and has_j and has_9:
                best_offer = (90, s, False, False)
            elif cnt >= 3 and has_j:
                best_offer = (80, s, False, False)

        # SA rules
        aces = sum(1 for c in self.hand if c.rank == 'A')
        tens = sum(1 for c in self.hand if c.rank == '10')
        if aces >= 4:
            best_offer = (100, 'SA', False, False)
        elif aces == 3:
            best_offer = (90, 'SA', False, False)
        elif aces >= 2 and tens >= 1 and best_offer is None:
            best_offer = (80, 'SA', False, False)

        # TA rules (use J/9 instead of A/10)
        jacks = sum(1 for c in self.hand if c.rank == 'J')
        nines = sum(1 for c in self.hand if c.rank == '9')
        if jacks >= 4:
            best_offer = (100, 'TA', False, False)
        elif jacks == 3:
            best_offer = (90, 'TA', False, False)
        elif jacks >= 2 and nines >= 1 and best_offer is None:
            best_offer = (80, 'TA', False, False)

        # More aggressive offers: estimate tricks roughly
        if best_offer is None:
            # estimate tricks: trumps + A + 10
            for s in SUITS:
                trumps = count_suit(self.hand, s)
                other_tricks = sum(1 for c in self.hand if c.rank in ('A','10') and c.suit != s)
                estimate = trumps + other_tricks
                if estimate >= 6:
                    return (110, s, False, False)

        return best_offer

    def _choose_defausse_suit(self):
        # pick weakest suit: minimal count, prefer singletons
        counts = {s: count_suit(self.hand, s) for s in SUITS}
        # choose suit with minimal count
        return min(counts.items(), key=lambda x: (x[1], x[0]))[0]

    def _rank_strength(self, card: Card, trump: str, lead_suit: Optional[str]) -> int:
        # higher means stronger
        if trump == 'TA' or card.suit == trump:
            try:
                return len(TRUMP_ORDER) - TRUMP_ORDER.index(card.rank)
            except ValueError:
                return 0
        else:
            try:
                return len(NORMAL_ORDER) - NORMAL_ORDER.index(card.rank)
            except ValueError:
                return 0

    def play_card(self, seat:int, leader:int, trick:list, trump:str):
        # Heuristic play (simplified from heuristiques.md)
        engine = getattr(self, 'engine', None)
        taker = getattr(engine, 'taker_idx', None)
        attacker = (taker is not None) and (taker % 2 == self.seat % 2)

        if not trick:
            # opening lead
            if attacker and engine and taker == self.seat:
                # preneur leads: play valet of trump if present to draw
                if has_rank(self.hand, trump, 'J'):
                    choice = next(c for c in self.hand if c.suit==trump and c.rank=='J')
                else:
                    # lead highest trump if have, else play master outside
                    tr = [c for c in self.hand if c.suit==trump]
                    if tr:
                        choice = max(tr, key=lambda c: self._rank_strength(c, trump, None))
                    else:
                        masters = [c for c in self.hand if c.rank in ('A','10')]
                        if masters:
                            choice = max(masters, key=lambda c: self._rank_strength(c, trump, None))
                        else:
                            # play smallest
                            choice = min(self.hand, key=lambda c: self._rank_strength(c, trump, None))
            else:
                # non-preneur or no contract: follow general rules
                masters = [c for c in self.hand if c.rank in ('A','10')]
                if masters:
                    choice = max(masters, key=lambda c: self._rank_strength(c, trump, None))
                else:
                    choice = min(self.hand, key=lambda c: self._rank_strength(c, trump, None))
        else:
            lead = trick[0][1].suit
            same = [c for c in self.hand if c.suit==lead]
            # find current winner of trick
            current_winner = trick[0]
            for t in trick[1:]:
                if self._rank_strength(t[1], trump, lead) > self._rank_strength(current_winner[1], trump, lead):
                    current_winner = t

            if same:
                # if can beat current, play smallest that wins, else smallest
                winning = [c for c in same if self._rank_strength(c, trump, lead) > self._rank_strength(current_winner[1], trump, lead)]
                if winning:
                    choice = min(winning, key=lambda c: self._rank_strength(c, trump, lead))
                else:
                    choice = min(same, key=lambda c: self._rank_strength(c, trump, lead))
            else:
                trumps = [c for c in self.hand if c.suit==trump]
                if trumps:
                    # if partner is winning, dump smallest; else try to overtrump
                    partner_idx = (self.seat + 2) % 4
                    partner_winning = (current_winner[0] == partner_idx)
                    if partner_winning:
                        choice = min(self.hand, key=lambda c: self._rank_strength(c, trump, None))
                    else:
                        higher_trumps = [c for c in trumps if self._rank_strength(c, trump, None) > self._rank_strength(current_winner[1], trump, lead)]
                        if higher_trumps:
                            choice = min(higher_trumps, key=lambda c: self._rank_strength(c, trump, None))
                        else:
                            choice = min(self.hand, key=lambda c: self._rank_strength(c, trump, None))
                else:
                    # no trump: discard worst card from weakest suit
                    s = self._choose_defausse_suit()
                    cand = [c for c in self.hand if c.suit==s]
                    if cand:
                        choice = min(cand, key=lambda c: self._rank_strength(c, trump, None))
                    else:
                        choice = min(self.hand, key=lambda c: self._rank_strength(c, trump, None))

        self.hand.remove(choice)
        return choice

class RLPlayer(Player):
    def __init__(self, name:str, policy=None):
        super().__init__(name)
        self.policy = policy

    def bid(self, current_best):
        return None

    def play_card(self, seat:int, leader:int, trick:list, trump:str):
        # expect policy to accept list of cards and return index
        if self.policy is None:
            return RandomPlayer.play_card(self, leader, trick, trump)
        idx = self.policy.choose_card(self.hand, leader, trick, trump)
        card = self.hand.pop(idx)
        return card


def create_player(strategy: str, name: str):
    s = strategy.lower()
    if s == 'random':
        return RandomPlayer(name)
    if s.startswith('heuristic') or s == 'user':
        # allow variants like 'heuristic:user' or 'heuristic:assistant'
        parts = s.split(':')
        variant = 'user' if len(parts) == 1 else parts[1]
        return HeuristicPlayer(name, variant=variant)
    if s == 'rl' or s == 'agent':
        return RLPlayer(name)
    # default
    return RandomPlayer(name)
