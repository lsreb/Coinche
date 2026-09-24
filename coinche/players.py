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
        self.initial_hand: List[Card] = list(hand)
        self._raise_contributions = set()

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

    # ---------------------------------------------------------------
    # Bidding (section 1 of heuristiques.md)
    # ---------------------------------------------------------------

    def _last_bid_info(self):
        """Returns (seat, bid) of the last bid actually made in the current auction."""
        engine = getattr(self, 'engine', None)
        if engine is None or not getattr(engine, 'history', None):
            return None
        for entry in reversed(engine.history.get('auction', [])):
            if entry.get('offer') is not None:
                return entry['seat'], entry['offer']
        return None

    def _count_tricks(self, trump_suit: Optional[str], hi: str = 'A', lo: str = '10', third: str = 'K') -> int:
        # 1 trick per trump card (if trump_suit given), then in each suit: 1 for
        # the master (hi), 1 for the second (lo) if it's the third card, 2 for
        # lo+third as the third card, 2 for hi+lo in the same suit. For SA (no
        # trump, hi='A'/lo='10'/third='K') and TA (all suits count,
        # hi='J'/lo='9'/third='A') this is called with trump_suit=None: no suit
        # is then excluded from the loop.
        tricks = count_suit(self.hand, trump_suit) if trump_suit else 0
        for s in SUITS:
            if s == trump_suit:
                continue
            cards = [c for c in self.hand if c.suit == s]
            ranks = {c.rank for c in cards}
            if hi in ranks and lo in ranks:
                tricks += 2
            elif lo in ranks and third in ranks and len(cards) >= 3:
                tricks += 2
            elif hi in ranks:
                tricks += 1
            elif lo in ranks and len(cards) >= 3:
                tricks += 1
        return tricks

    def _decrochage_ok(self, own_level: int, opp_level: int) -> bool:
        # A "décrochage" (competitive overcall) tolerates a "small lie" of one
        # level (10 points) beyond what the hand truly justifies: a 100-hand
        # can say 110 when the opponent said 100, but an 80-hand can't jump to 110.
        return own_level >= opp_level - 10

    def _my_last_bid(self):
        """Returns my own last offer made in the current auction (or None)."""
        engine = getattr(self, 'engine', None)
        seat = getattr(self, 'seat', None)
        if engine is None or not getattr(engine, 'history', None):
            return None
        for entry in reversed(engine.history.get('auction', [])):
            if entry['seat'] == seat and entry.get('offer') is not None:
                return entry['offer']
        return None

    def _last_partner_bid(self):
        """My partner's last offer in the current auction, even if an opponent
        has bid since (unlike _last_bid_info, which only looks at the very
        last offer of the auction)."""
        engine = getattr(self, 'engine', None)
        seat = getattr(self, 'seat', None)
        if engine is None or not getattr(engine, 'history', None) or seat is None:
            return None
        partner_seat = (seat + 2) % 4
        for entry in reversed(engine.history.get('auction', [])):
            if entry['seat'] == partner_seat and entry.get('offer') is not None:
                return entry['offer']
        return None

    def _team_last_bidder_is_me(self) -> bool:
        # True if, between my partner and me, I am the last one to have
        # actually bid (not passed) in this auction -- including if my
        # partner never bid at all. Used to forbid décrochage (heuristiques.md
        # §1.1) when my partner hasn't given me any new information since my
        # own last bid.
        engine = getattr(self, 'engine', None)
        seat = getattr(self, 'seat', None)
        if engine is None or not getattr(engine, 'history', None) or seat is None:
            return False
        partner_seat = (seat + 2) % 4
        for entry in reversed(engine.history.get('auction', [])):
            if entry.get('offer') is None:
                continue
            if entry['seat'] == seat:
                return True
            if entry['seat'] == partner_seat:
                return False
        return False

    def _decrochage_adjusted_level(self, bidder_seat, bid):
        # If `bidder_seat`'s `bid` immediately follows an opposing offer at a
        # lower level, it may itself be a décrochage: it then represents a
        # real level 10 lower than what it announces. We look up the exact
        # bid in the history (not just the previous one), since this function
        # can be called well after the fact, on a past bid.
        level = bid[0]
        engine = getattr(self, 'engine', None)
        if engine is not None and getattr(engine, 'history', None):
            made = [(e['seat'], tuple(e['offer'])) for e in engine.history.get('auction', []) if e.get('offer') is not None]
            for i, (seat, offer) in enumerate(made):
                if seat == bidder_seat and offer == tuple(bid):
                    if i >= 1:
                        prev_seat, prev_offer = made[i - 1]
                        if prev_seat % 2 != bidder_seat % 2 and prev_offer[0] == level - 10:
                            return level - 10
                    break
        return level

    def _sa_known_aces(self, bidder_seat, bid):
        aces_table = {80: 2, 90: 3, 100: 4}
        return aces_table.get(self._decrochage_adjusted_level(bidder_seat, bid), 0)

    def _my_earlier_bid_of_type(self, bid_type):
        """My first offer of this type (SA/TA) in the current auction, if I
        made one before my partner's current offer (a sign that their raise
        already includes the information I revealed myself, not to be
        recounted)."""
        engine = getattr(self, 'engine', None)
        seat = getattr(self, 'seat', None)
        if engine is None or not getattr(engine, 'history', None) or seat is None:
            return None
        for entry in engine.history.get('auction', []):
            offer = entry.get('offer')
            if entry['seat'] == seat and offer is not None and offer[1] == bid_type:
                return offer
        return None

    def _partner_sa_to_color(self, partner_seat, partner_bid, own_color_candidates):
        # We can switch to our own suit rather than simply supporting
        # partner's SA, relying on the aces already revealed by their SA bid
        # (minus one, potentially at the chosen trump suit): see
        # heuristiques.md §1.1.
        if not own_color_candidates or 'sa_partner_credit' in self._raise_contributions:
            return None
        known_aces = self._sa_known_aces(partner_seat, partner_bid)
        credited_aces = known_aces - 1
        if credited_aces <= 0:
            return None
        level, trump = max(own_color_candidates, key=lambda c: c[0])[:2]
        self._raise_contributions.add('sa_partner_credit')
        return (level + 10 * credited_aces, trump, False, False)

    def _partner_color_remonte(self, partner_bid):
        # Each type of information (trump support, outside ace) is only
        # revealed once per donne: two partners shouldn't keep re-raising on
        # the same card, but can raise on different rounds for different
        # reasons.
        level, trump = partner_bid[0], partner_bid[1]
        if trump in ('SA', 'TA'):
            return None
        contrib = self._raise_contributions
        my_trumps = count_suit(self.hand, trump)
        has_9_second = has_rank(self.hand, trump, '9') and my_trumps >= 2
        has_3_trumps = my_trumps >= 3
        new_level = level
        gave_support = False
        if level == 80 and 'color_support' not in contrib and (has_9_second or has_3_trumps):
            new_level += 10
            gave_support = True
        elif level == 80 and 'color_support' not in contrib:
            return None
        jack_nine_known = level >= 90 or has_9_second or gave_support
        if jack_nine_known and my_trumps >= 1 and 'color_exter' not in contrib:
            exter_aces = sum(1 for c in self.hand if c.rank == 'A' and c.suit != trump)
            my_last_bid = self._my_last_bid()
            if exter_aces > 0 and my_last_bid is not None and my_last_bid[1] == 'SA':
                # My partner switched to this suit (_partner_sa_to_color)
                # based on my SA: they already credited (assumed aces - 1) of
                # my aces. I should only add what's left, not recount my aces
                # minus one.
                seat = getattr(self, 'seat', None)
                assumed = self._sa_known_aces(seat, my_last_bid)
                already_credited = max(0, assumed - 1)
                exter_aces = max(0, exter_aces - already_credited)
            if exter_aces > 0:
                new_level += 10 * exter_aces
                contrib.add('color_exter')
        if gave_support:
            contrib.add('color_support')
        return (new_level, trump, False, False) if new_level > level else None

    def _combined_count(self, partner_bid, bid_type, my_count):
        # How many master cards (aces for SA, jacks for TA) does the team
        # actually have, combining the partner's bid (décrochage included)
        # and my own? If the partner's raise itself comes from my own
        # earlier bid of the same type, my cards are already counted there:
        # no double counting.
        table = {80: 2, 90: 3, 100: 4}
        partner_seat = (getattr(self, 'seat', 0) + 2) % 4
        genuine_level = self._decrochage_adjusted_level(partner_seat, partner_bid)
        base = table.get(genuine_level, 0)
        if self._my_earlier_bid_of_type(bid_type) is not None:
            return base
        return base + my_count

    def _partner_sa_remonte(self, partner_bid):
        level = partner_bid[0]
        contrib = self._raise_contributions
        my_aces = sum(1 for c in self.hand if c.rank == 'A')
        new_level = level
        if my_aces > 0 and 'sa_aces' not in contrib:
            new_level += 10 * my_aces
            contrib.add('sa_aces')
        combined_aces = self._combined_count(partner_bid, 'SA', my_aces)
        # As with suit contracts, holding all the aces isn't enough to bid
        # the non-third tens on top: at least 4 tricks must actually be
        # counted in your own hand (same method as the suit décrochage),
        # otherwise we'd just be stacking bonuses unrelated to real tricks.
        # And a partner who has no ace themselves doesn't raise at all, even
        # for this bonus: it's not their place to speak for others' aces.
        if my_aces > 0 and combined_aces >= 4 and 'sa_tens' not in contrib and self._count_tricks(None) >= 4:
            non_sec_tens = sum(1 for s in SUITS if has_rank(self.hand, s, '10') and count_suit(self.hand, s) >= 2)
            if non_sec_tens > 0:
                new_level += 10 * non_sec_tens
                contrib.add('sa_tens')
        return (new_level, 'SA', False, False) if new_level > level else None

    def _partner_ta_remonte(self, partner_bid):
        level = partner_bid[0]
        contrib = self._raise_contributions
        my_jacks = sum(1 for c in self.hand if c.rank == 'J')
        new_level = level
        if my_jacks > 0 and 'ta_jacks' not in contrib:
            new_level += 10 * my_jacks
            contrib.add('ta_jacks')
        combined_jacks = self._combined_count(partner_bid, 'TA', my_jacks)
        # Same guard as SA: all the jacks combined isn't enough, we also need
        # at least 4 tricks actually counted (jack/9/ace instead of
        # ace/10/king), and a partner with no jack of their own doesn't raise
        # at all.
        if my_jacks > 0 and combined_jacks >= 4 and 'ta_nines' not in contrib and self._count_tricks(None, hi='J', lo='9', third='A') >= 4:
            non_sec_nines = sum(1 for s in SUITS if has_rank(self.hand, s, '9') and count_suit(self.hand, s) >= 2)
            if non_sec_nines > 0:
                new_level += 10 * non_sec_nines
                contrib.add('ta_nines')
        return (new_level, 'TA', False, False) if new_level > level else None

    def bid(self, current_best):
        seat = getattr(self, 'seat', None)
        last = self._last_bid_info()
        partner_bid = None
        opponent_bid = None
        if current_best is not None and last is not None and seat is not None and last[0] != seat:
            if last[0] % 2 == seat % 2:
                partner_bid = current_best
            else:
                opponent_bid = current_best
                # An opponent bid after my partner: I can still raise my
                # partner's last bid, not just décrocher (overcall) on my own
                # hand (heuristiques.md §1.1, TA/SA/suit note).
                earlier_partner_bid = self._last_partner_bid()
                if earlier_partner_bid is not None:
                    partner_bid = earlier_partner_bid

        candidates = []

        # 1.1) Suit
        for s in SUITS:
            cnt = count_suit(self.hand, s)
            has_j = has_rank(self.hand, s, 'J')
            has_9 = has_rank(self.hand, s, '9')
            if cnt >= 4 and has_j and has_9:
                tricks = self._count_tricks(s)
                if tricks >= 8:
                    candidates.append((250, s, False, True))
                elif tricks >= 5:
                    candidates.append((tricks * 10 + 60, s, False, False))
                else:
                    candidates.append((100, s, False, False))
            elif cnt >= 3 and has_j and has_9:
                candidates.append((90, s, False, False))
            elif cnt >= 3 and has_j:
                candidates.append((80, s, False, False))

        # 1.2) SA
        aces = sum(1 for c in self.hand if c.rank == 'A')
        tens_non_sec = sum(1 for s in SUITS if has_rank(self.hand, s, '10') and count_suit(self.hand, s) >= 2)
        sa_candidate = None
        if aces >= 4:
            sa_candidate = (100, 'SA', False, False)
        elif aces == 3:
            sa_candidate = (90, 'SA', False, False)
        elif aces >= 2 and tens_non_sec >= 1:
            sa_candidate = (80, 'SA', False, False)
        if sa_candidate is not None:
            candidates.append(sa_candidate)

        # 1.3) TA (jacks instead of aces, 9s instead of 10s)
        jacks = sum(1 for c in self.hand if c.rank == 'J')
        nines_non_sec = sum(1 for s in SUITS if has_rank(self.hand, s, '9') and count_suit(self.hand, s) >= 2)
        ta_candidate = None
        if jacks >= 4:
            ta_candidate = (100, 'TA', False, False)
        elif jacks == 3:
            ta_candidate = (90, 'TA', False, False)
        elif jacks >= 2 and nines_non_sec >= 1:
            ta_candidate = (80, 'TA', False, False)
        if ta_candidate is not None:
            candidates.append(ta_candidate)

        # Partner's raise
        if partner_bid is not None:
            if partner_bid[1] == 'SA':
                r = self._partner_sa_remonte(partner_bid)
                if r is not None:
                    candidates.append(r)
                own_color_candidates = [c for c in candidates if c[1] in SUITS]
                partner_seat = (seat + 2) % 4
                r2 = self._partner_sa_to_color(partner_seat, partner_bid, own_color_candidates)
                if r2 is not None:
                    candidates.append(r2)
            elif partner_bid[1] == 'TA':
                r = self._partner_ta_remonte(partner_bid)
                if r is not None:
                    candidates.append(r)
            else:
                r = self._partner_color_remonte(partner_bid)
                if r is not None:
                    candidates.append(r)

        # Highest bid among all available suits/types; at an equal level, we
        # prefer extending the partner's bid over a "solo" bid.
        def _bid_key(b):
            follows_partner = partner_bid is not None and b[1] == partner_bid[1]
            return (b[0], 1 if follows_partner else 0)
        best_offer = max(candidates, key=_bid_key) if candidates else None

        # Our own aces/jacks are already "announced" as soon as we make our
        # initial bid (the 80/90/100 level encodes them): they must not be
        # recounted later as a raise, or the contract would be artificially
        # inflated.
        if best_offer is not None and best_offer == sa_candidate:
            self._raise_contributions.add('sa_aces')
        if best_offer is not None and best_offer == ta_candidate:
            self._raise_contributions.add('ta_jacks')

        # Décrochage: only in response to an opponent (never to the
        # partner), and only if the hand is no more than one level short of
        # the opponent's. If I am myself the last of my team to have
        # actually bid (my partner hasn't given me any new information
        # since), I don't décrocher either (heuristiques.md §1.1): I mustn't
        # raise on my own beyond what my own hand already justified.
        if opponent_bid is not None and best_offer is not None and not self._team_last_bidder_is_me():
            opp_level = opponent_bid[0]
            if opp_level <= 100 and best_offer[0] <= opp_level:
                next_level = opp_level + 10
                trump = best_offer[1]
                can_decrochage = next_level <= 110 and self._decrochage_ok(best_offer[0], opp_level)
                if can_decrochage and next_level >= 100 and trump in SUITS:
                    # Without certainty that the partner holds the 9, a
                    # décrochage up to 100 is only credible if we count at
                    # least 4 tricks in our own hand (any outside ace/ten is
                    # already included in this count).
                    can_decrochage = self._count_tricks(trump) >= 4
                if can_decrochage:
                    best_offer = (next_level, trump, best_offer[2], best_offer[3])
                    if next_level >= 100 and trump in SUITS:
                        # The trump support and any outside ace already
                        # served to justify this décrochage: they must not be
                        # re-announced as a further raise when the partner
                        # raises again.
                        self._raise_contributions.add('color_support')
                        self._raise_contributions.add('color_exter')

        # Coinche (heuristiques.md §1.4): only on defense (the opponent has
        # the best bid), and only if we don't otherwise have a hand that
        # would justify outbidding instead.
        if opponent_bid is not None and (best_offer is None or best_offer[0] < opponent_bid[0] + 10):
            if self._can_coincher(opponent_bid):
                return (opponent_bid[0], opponent_bid[1], True, opponent_bid[3])

        return best_offer

    def _needed_tricks_to_coincher(self, level: int) -> int:
        # The higher the opponent's contract, the fewer tricks the defense
        # needs to justify coinching (the taker has that much more trouble
        # succeeding): 2 tricks at 130, 3 at 120, 4 at 110, etc.
        # (heuristiques.md §1.4).
        return max(0, (150 - level) // 10)

    def _defense_trump_tricks(self, trump: str) -> int:
        # Unlike the attacker (1 trick per trump held, §1.1, since they
        # control the suit), the defense can't assume its trumps hold up on
        # their own (heuristiques.md §1.4): any 3 trumps yield no trick by
        # default. The ace as the third trump (or more) is worth 1 trick
        # overall; the ace and the 10 with two more trumps (4 total) are
        # worth 2 tricks.
        trumps = [c for c in self.hand if c.suit == trump]
        ranks = {c.rank for c in trumps}
        if 'A' in ranks and '10' in ranks and len(trumps) >= 4:
            return 2
        if 'A' in ranks and len(trumps) >= 3:
            return 1
        return 0

    def _can_coincher(self, opponent_bid) -> bool:
        level, trump, _coinched, _capot = opponent_bid
        needed = self._needed_tricks_to_coincher(level)
        if trump in SUITS:
            if count_suit(self.hand, trump) < 3:
                return False
            off_trump_tricks = self._count_tricks(trump) - count_suit(self.hand, trump)
            return self._defense_trump_tricks(trump) + off_trump_tricks >= needed
        if trump == 'SA':
            return self._count_tricks(None) >= needed
        if trump == 'TA':
            return self._count_tricks(None, hi='J', lo='9', third='A') >= needed
        return False

    # ---------------------------------------------------------------
    # Card play (section 2 of heuristiques.md)
    # ---------------------------------------------------------------

    def _rank_strength(self, card: Card, trump: str, lead_suit: Optional[str]):
        # (is_trump, is_lead, -rank_idx): same logic as GameEngine.card_order_key,
        # so a trump always beats an off-suit card when comparing across suits.
        is_trump = 1 if (trump == 'TA' or card.suit == trump) else 0
        order = TRUMP_ORDER if is_trump else NORMAL_ORDER
        try:
            rank_idx = order.index(card.rank)
        except ValueError:
            rank_idx = 99
        is_lead = 1 if (lead_suit is not None and card.suit == lead_suit) else 0
        return (is_trump, is_lead, -rank_idx)

    def _master_ranks(self, trump: str):
        # Under TA, all suits count as trump: jack then 9.
        return ('J', '9') if trump == 'TA' else ('A', '10')

    def _weakest(self, cards, trump: str, lead: Optional[str] = None) -> Card:
        return min(cards, key=lambda c: self._rank_strength(c, trump, lead))

    def _strongest(self, cards, trump: str, lead: Optional[str] = None) -> Card:
        return max(cards, key=lambda c: self._rank_strength(c, trump, lead))

    def _long_suit_lead(self, cards, trump: str, master_lo: str) -> Card:
        # Long suit without ace/jack, holding the 10/9 (heuristiques.md
        # §2.1.2/§2.1.3): we keep the 10/9 and play the worst card of the
        # contiguous run that immediately follows it in rank order (e.g.
        # 10-King-Queen-7 -> Queen, keeping the 10 and the 7). If there's no
        # contiguous card, we fall back to the lowest card.
        order = TRUMP_ORDER if trump == 'TA' else NORMAL_ORDER
        ranks_held = {c.rank for c in cards}
        idx = order.index(master_lo) + 1
        worst_adjacent_rank = None
        while idx < len(order) and order[idx] in ranks_held:
            worst_adjacent_rank = order[idx]
            idx += 1
        if worst_adjacent_rank is not None:
            return next(c for c in cards if c.rank == worst_adjacent_rank)
        return self._weakest(cards, trump)

    def _played_cards(self):
        """Cards already played in the completed tricks of the current donne."""
        engine = getattr(self, 'engine', None)
        if engine is None or not getattr(engine, 'history', None):
            return []
        return [p['card'] for t in engine.history.get('tricks', []) for p in t['plays']]

    def _card_seen_before(self, suit: str, rank: str) -> bool:
        return f"{rank}{suit}" in self._played_cards()

    def _trumps_remain_with_opponents(self, trump: str) -> bool:
        seen = sum(1 for c in self.hand if c.suit == trump)
        seen += sum(1 for card in self._played_cards() if card[-1] == trump)
        return seen < 8

    def _trump_already_led(self, trump: str) -> bool:
        # Holding back the "second 9" only applies to the very first trump
        # round: on the next round the rule reverses (play the 9 if you have it).
        return any(card[-1] == trump for card in self._played_cards())

    def _attaquant_principal_seat(self):
        # heuristiques.md distinguishes the "main taker"/"main attacker" (the
        # one who first bid the type of bid ultimately retained, e.g. the
        # initial 90 Hearts) from the "following partner" who then raised
        # (e.g. the 120 Hearts) -- §2.1.1.1, §2.1.3, §2.5. This isn't
        # necessarily the same seat as engine.taker_idx, which is only the
        # author of the very last retained bid (correct for scoring and for
        # "the taker" in the coinche sense, §1.4, but not for this play
        # distinction between the two attackers).
        engine = getattr(self, 'engine', None)
        taker = getattr(engine, 'taker_idx', None)
        contract = getattr(engine, 'contract', None)
        if engine is None or taker is None or contract is None or not getattr(engine, 'history', None):
            return taker
        team = taker % 2
        for entry in engine.history.get('auction', []):
            offer = entry.get('offer')
            if offer is not None and entry['seat'] % 2 == team and offer[1] == contract.trump:
                return entry['seat']
        return taker

    def _color_contract_coinched(self) -> bool:
        # heuristiques.md §2.5 only applies to a coinched suit contract (not SA/TA).
        engine = getattr(self, 'engine', None)
        contract = getattr(engine, 'contract', None)
        return bool(contract is not None and contract.coinched and contract.trump in SUITS)

    def _coincheur_seat(self):
        # The seat that actually called the coinche, if the auction ended
        # that way (last offer of the auction with the coinche flag).
        engine = getattr(self, 'engine', None)
        if engine is None or not getattr(engine, 'history', None):
            return None
        auction = engine.history.get('auction', [])
        if not auction:
            return None
        last = auction[-1]
        offer = last.get('offer')
        return last['seat'] if offer is not None and offer[2] else None

    def _is_coincheur_before_taker(self) -> bool:
        # heuristiques.md §2.5: the defender who called the coinche
        # themselves, and who sits right before the MAIN taker (the
        # initiator, see _attaquant_principal_seat, not necessarily
        # engine.taker_idx) in play order, gives absolute priority to
        # leading their long suit.
        if not self._color_contract_coinched():
            return False
        preneur_principal = self._attaquant_principal_seat()
        seat = getattr(self, 'seat', None)
        if preneur_principal is None or seat is None or seat != (preneur_principal + 3) % 4:
            return False
        return self._coincheur_seat() == seat

    def _is_confirmed_master(self, card: Card, trump: str) -> bool:
        # True even if it's neither the ace nor the 10: every card that
        # outranks it in its suit has already been played, so it's
        # guaranteed to win.
        order = TRUMP_ORDER if (trump == 'TA' or card.suit == trump) else NORMAL_ORDER
        higher_ranks = order[:order.index(card.rank)]
        return all(self._card_seen_before(card.suit, r) for r in higher_ranks)

    def _choose_attack_suit(self, trump):
        # The attack wants to lead in the suit where it initially held the
        # most cards (heuristiques.md §2.1.1.2), not just play the weakest
        # card across all suits.
        initial = getattr(self, 'initial_hand', self.hand)
        lengths = {s: sum(1 for c in initial if c.suit == s) for s in SUITS if s != trump}
        available = [s for s in lengths if count_suit(self.hand, s) > 0]
        if not available:
            return None
        return max(available, key=lambda s: lengths[s])

    def _suit_to_replay_for_partner(self, master_hi):
        # If my partner made their very first lead in a suit and I won it
        # with my master card (the ace in SA, the jack in TA), I must lead
        # that suit again as a priority as long as I still hold the master
        # there (heuristiques.md §2.1.2/§2.1.3).
        engine = getattr(self, 'engine', None)
        seat = getattr(self, 'seat', None)
        if engine is None or not getattr(engine, 'history', None) or seat is None:
            return None
        partner_idx = (seat + 2) % 4
        partner_led_suits = set()
        for trick in engine.history.get('tricks', []):
            plays = trick.get('plays', [])
            if not plays or plays[0]['seat'] != partner_idx:
                continue
            lead_suit = plays[0]['card'][-1]
            if lead_suit in partner_led_suits:
                continue
            partner_led_suits.add(lead_suit)
            my_play = next((p['card'] for p in plays if p['seat'] == seat), None)
            if my_play and my_play[:-1] == master_hi and my_play[-1] == lead_suit:
                return lead_suit
        return None

    def _lead_offsuit_master(self, trump, master_hi, master_lo):
        offsuit = [c for c in self.hand if c.suit != trump]
        if not offsuit:
            return self._weakest(self.hand, trump)
        masters = [c for c in offsuit if c.rank == master_hi]
        if masters:
            return masters[0]
        seconds = [c for c in offsuit if c.rank == master_lo and self._card_seen_before(c.suit, master_hi)]
        if seconds:
            return seconds[0]
        confirmed = [c for c in offsuit if self._is_confirmed_master(c, trump)]
        if confirmed:
            return self._strongest(confirmed, trump)
        suit = self._choose_attack_suit(trump)
        pool = [c for c in offsuit if c.suit == suit] if suit else offsuit
        return self._weakest(pool, trump)

    def _lead_coinche_defense_long_suit(self, trump, master_hi, master_lo):
        # heuristiques.md §2.5: aggressive defense by the coincheur right
        # before the main taker, absolute priority to the long suit
        # (starting with the ace if it's there) to force them to cut. The
        # attack, on the other hand, plays normally when it gets coinched (§2.5).
        suit = self._choose_attack_suit(trump)
        if suit is None:
            return self._weakest(self.hand, trump)
        cards = [c for c in self.hand if c.suit == suit]
        aces = [c for c in cards if c.rank == master_hi]
        if aces:
            return aces[0]
        return self._weakest(cards, trump)

    def _lead_taker_trump(self, trump, master_hi, master_lo):
        if has_rank(self.hand, trump, 'J'):
            return next(c for c in self.hand if c.suit == trump and c.rank == 'J')
        trumps = [c for c in self.hand if c.suit == trump]
        if trumps:
            if self._trump_already_led(trump) and not self._trumps_remain_with_opponents(trump):
                # no trump remains with the opponents: no point playing our
                # last trump (we know we're the only one left with any),
                # better to keep it for a possible cut and attack elsewhere.
                return self._lead_offsuit_master(trump, master_hi, master_lo)
            # second 9: only on the very first trump round, we keep the 9
            # (our highest trump, lacking the jack) for the next round.
            if (not self._trump_already_led(trump)) and has_rank(self.hand, trump, '9') and len(trumps) > 1:
                others = [c for c in trumps if c.rank != '9']
                return self._weakest(others, trump)
            if has_rank(self.hand, trump, '9'):
                return next(c for c in trumps if c.rank == '9')
            if self._trumps_remain_with_opponents(trump):
                return self._weakest(trumps, trump)
        return self._lead_offsuit_master(trump, master_hi, master_lo)

    def _lead_partner_trump(self, trump, master_hi, master_lo):
        trumps = [c for c in self.hand if c.suit == trump]
        if trumps:
            if self._trump_already_led(trump) and not self._trumps_remain_with_opponents(trump):
                # same reason as on the main attacker's side: don't waste
                # the last trump when we know we're the only one still
                # holding any.
                return self._lead_offsuit_master(trump, master_hi, master_lo)
            best = self._strongest(trumps, trump)
            if best.rank == '9' and len(trumps) > 1 and not self._trump_already_led(trump):
                # same logic as the jack round: we keep the 9 for the next round
                others = [c for c in trumps if c.rank != '9']
                return self._weakest(others, trump)
            return best
        return self._lead_offsuit_master(trump, master_hi, master_lo)

    def _lead_defense(self, trump, master_hi, master_lo):
        if self._is_coincheur_before_taker():
            # heuristiques.md §2.5: the coincheur right before the taker
            # gives absolute priority to their long suit, even before their
            # outside aces.
            return self._lead_coinche_defense_long_suit(trump, master_hi, master_lo)
        offsuit_masters = [c for c in self.hand if c.rank == master_hi and c.suit != trump]
        if offsuit_masters:
            return offsuit_masters[0]
        # Same logic as _lead_offsuit_master (attack side): the 10 having
        # become master (ace already played), then any card having become
        # master by elimination even if it's neither the ace nor the 10
        # (heuristiques.md line 71/128, _is_confirmed_master) -- previously,
        # only a literal ace was recognized as a master card to lead, a
        # queen or king that had become master by elimination fell into the
        # "weakest" fallback and was wasted as a discard instead of being led.
        offsuit = [c for c in self.hand if c.suit != trump]
        seconds = [c for c in offsuit if c.rank == master_lo and self._card_seen_before(c.suit, master_hi)]
        if seconds:
            return seconds[0]
        confirmed = [c for c in offsuit if self._is_confirmed_master(c, trump)]
        if confirmed:
            return self._strongest(confirmed, trump)
        singletons = [s for s in SUITS if s != trump and count_suit(self.hand, s) == 1]
        if singletons:
            return next(c for c in self.hand if c.suit == singletons[0])
        return self._weakest(self.hand, trump)

    def _prefer_9_second_suit(self, candidates, trump, is_taker):
        # In TA, the taker's partner (not the taker themselves) looks to
        # lead in the suit where they have a second (or later) 9, to let
        # their partner know they hold the 9 (heuristiques.md §2.1.3,
        # TA-specific).
        if trump != 'TA' or is_taker or len(candidates) < 2:
            return candidates
        preferred = [(s, cards) for s, cards in candidates
                     if has_rank(self.hand, s, '9') and count_suit(self.hand, s) >= 2]
        rest = [c for c in candidates if c not in preferred]
        return preferred + rest if preferred else candidates

    def _lead_attack_sa_ta(self, trump, master_hi, master_lo, is_taker=False):
        # Absolute priority: a suit that has become master by elimination
        # (the 10/9 after the ace/jack has been played) is led before
        # anything else, even before returning to the suit the partner
        # opened (heuristiques.md §2.1.2/§2.1.3). master_hi (ace/jack) is
        # always trivially "confirmed" (highest rank in its order): that's
        # not what's at stake here.
        confirmed_by_suit = {}
        for c in self.hand:
            if c.rank != master_hi and self._is_confirmed_master(c, trump):
                confirmed_by_suit.setdefault(c.suit, []).append(c)
        if confirmed_by_suit:
            suit = next(iter(confirmed_by_suit))
            return self._strongest(confirmed_by_suit[suit], trump)

        # Otherwise, if my partner made their first lead in a suit and I won
        # it with my master card, I lead that suit again as long as I still
        # have cards in it: no need for total certainty about the remaining
        # card, winning with the ace/jack is enough.
        replay_suit = self._suit_to_replay_for_partner(master_hi)
        if replay_suit is not None and count_suit(self.hand, replay_suit) > 0:
            replay_cards = [c for c in self.hand if c.suit == replay_suit]
            return self._strongest(replay_cards, trump)

        long_with_master, long_without_master, short_suits = [], [], []
        for s in SUITS:
            cards = [c for c in self.hand if c.suit == s]
            if not cards:
                continue
            if len(cards) >= 3 and any(c.rank == master_hi for c in cards):
                long_with_master.append((s, cards))
            elif len(cards) >= 3 and any(c.rank == master_lo for c in cards):
                # long suit without the ace, with a 10 (heuristiques.md §2.1.2)
                long_without_master.append((s, cards))
            else:
                short_suits.append((s, cards))
        long_without_master = self._prefer_9_second_suit(long_without_master, trump, is_taker)
        short_suits = self._prefer_9_second_suit(short_suits, trump, is_taker)

        if long_with_master:
            s, cards = long_with_master[0]
            if self._card_seen_before(s, master_hi) and any(c.rank == master_lo for c in cards):
                return next(c for c in cards if c.rank == master_lo)
            if len(long_with_master) == 1 and not long_without_master and not short_suits:
                return next(c for c in cards if c.rank == master_hi)
            # we avoid playing our master card on the first round of a suit if another lead is available
            if long_without_master:
                _, cards2 = long_without_master[0]
                return self._long_suit_lead(cards2, trump, master_lo)
            if short_suits:
                _, cards2 = short_suits[0]
                return self._strongest(cards2, trump)
            return next(c for c in cards if c.rank == master_hi)

        if long_without_master:
            s, cards = long_without_master[0]
            return self._long_suit_lead(cards, trump, master_lo)

        if short_suits:
            s, cards = short_suits[0]
            return self._strongest(cards, trump)

        return self._weakest(self.hand, trump)

    def _lead_card(self, trump, is_attacker, is_taker, master_hi, master_lo):
        if trump in SUITS:
            if is_attacker and is_taker:
                return self._lead_taker_trump(trump, master_hi, master_lo)
            if is_attacker:
                return self._lead_partner_trump(trump, master_hi, master_lo)
            return self._lead_defense(trump, master_hi, master_lo)
        if is_attacker:
            return self._lead_attack_sa_ta(trump, master_hi, master_lo, is_taker)
        return self._lead_defense(trump, master_hi, master_lo)

    def _is_absolute_master_rank(self, trump: str) -> str:
        return 'J' if (trump in SUITS or trump == 'TA') else 'A'

    def _partner_is_absolute_master(self, current_winner, trump) -> bool:
        partner_idx = (self.seat + 2) % 4
        if current_winner[0] != partner_idx:
            return False
        card = current_winner[1]
        master_rank = self._is_absolute_master_rank(trump)
        if trump in SUITS:
            # in a suit contract, only the trump jack is truly unstoppable
            # (an off-trump master card can always be cut by an opponent
            # still holding trump)
            return card.suit == trump and card.rank == master_rank
        return card.rank == master_rank

    def _choose_defausse_suit(self):
        # weak suit = an initial singleton, or an initial doubleton with no 10 or ace
        initial = getattr(self, 'initial_hand', self.hand)
        weak = []
        for s in SUITS:
            initial_cards = [c for c in initial if c.suit == s]
            if len(initial_cards) in (1, 2) and not any(c.rank in ('A', '10') for c in initial_cards):
                weak.append(s)
        candidates = [s for s in weak if count_suit(self.hand, s) > 0]
        if not candidates:
            candidates = [s for s in SUITS if count_suit(self.hand, s) > 0]
        return random.choice(candidates) if candidates else SUITS[0]

    def _discard(self, trump, current_winner):
        s = self._choose_defausse_suit()
        cand = [c for c in self.hand if c.suit == s] or self.hand
        if self._partner_is_absolute_master(current_winner, trump):
            master_rank = self._is_absolute_master_rank(trump)
            non_master = [c for c in cand if c.rank != master_rank]
            pool = non_master or cand
            return self._strongest(pool, trump)
        return self._weakest(cand, trump)

    def _secure_when_partner_wins(self, same, trump, lead, trick):
        # The trick is already won for my side: I choose which of my cards
        # in this suit to play now (risk-free), and which to keep in hand.
        order = TRUMP_ORDER if (trump == 'TA' or lead == trump) else NORMAL_ORDER
        ordered = sorted(same, key=lambda c: self._rank_strength(c, trump, lead), reverse=True)
        top = ordered[0]
        higher_ranks = order[:order.index(top.rank)]
        top_secured = all(
            self._card_seen_before(top.suit, r) or any(c.suit == top.suit and c.rank == r for _, c in trick)
            for r in higher_ranks
        )
        if top_secured or len(ordered) == 1:
            # my best card will become master later: I keep it
            return ordered[-1]
        if len(ordered) == 2:
            # jack/king as the second card: I secure it now rather than the low card
            return ordered[0]
        # 3 cards or more: I secure the middle card, keeping the best in reserve
        return ordered[1]

    def _follow_card(self, trick, trump, is_attacker, master_hi, master_lo):
        lead = trick[0][1].suit
        same = [c for c in self.hand if c.suit == lead]
        current_winner = trick[0]
        for t in trick[1:]:
            if self._rank_strength(t[1], trump, lead) > self._rank_strength(current_winner[1], trump, lead):
                current_winner = t

        if same:
            if is_attacker and lead == trump and trump in SUITS:
                # Continuing "drawing trumps" (2.1.1.1): jack then 9 (always
                # legal here: nothing beats the jack, and if we hold the 9
                # without the jack, either it beats the current master, or
                # that master is the jack itself and nothing can overtrump
                # it anyway - regles_coinche.md §4). Without either, however,
                # we must check whether some card is required to overtrump
                # the current master before playing the weakest: otherwise
                # we risk illegally undertrumping.
                for rank in ('J', '9'):
                    for c in same:
                        if c.rank == rank:
                            return c
                beating = [c for c in same
                           if self._rank_strength(c, trump, lead) > self._rank_strength(current_winner[1], trump, lead)]
                return self._weakest(beating, trump, lead) if beating else self._weakest(same, trump, lead)
            winning = [c for c in same if self._rank_strength(c, trump, lead) > self._rank_strength(current_winner[1], trump, lead)]
            if winning:
                # Priority to the master card (the ace, or the 10 if the ace
                # has already been played) to become master of the trick,
                # otherwise the smallest card that wins the trick.
                masters = [c for c in winning if c.rank == master_hi]
                if masters:
                    return masters[0]
                seconds = [c for c in winning if c.rank == master_lo and self._card_seen_before(lead, master_hi)]
                if seconds:
                    return seconds[0]
                return self._weakest(winning, trump, lead)
            partner_idx = (self.seat + 2) % 4
            if current_winner[0] == partner_idx and (
                self._partner_is_absolute_master(current_winner, trump) or len(trick) == 3
            ):
                return self._secure_when_partner_wins(same, trump, lead, trick)
            return self._weakest(same, trump, lead)

        trumps = [c for c in self.hand if c.suit == trump] if trump in SUITS else []
        if trumps:
            partner_idx = (self.seat + 2) % 4
            if current_winner[0] == partner_idx:
                return self._weakest(self.hand, trump)
            higher_trumps = [c for c in trumps if self._rank_strength(c, trump, lead) >
                              self._rank_strength(current_winner[1], trump, lead)]
            if higher_trumps:
                if is_attacker:
                    # attack: economy, the smallest card that wins
                    return self._weakest(higher_trumps, trump)
                # defense: cut with the highest trump, except a third 9 or a fourth ace
                nine_third = has_rank(self.hand, trump, '9') and len(trumps) == 3
                as_fourth = has_rank(self.hand, trump, 'A') and len(trumps) == 4
                if nine_third or as_fourth:
                    return self._weakest(higher_trumps, trump)
                return self._strongest(higher_trumps, trump)
            return self._weakest(self.hand, trump)

        return self._discard(trump, current_winner)

    def play_card(self, seat:int, leader:int, trick:list, trump:str):
        engine = getattr(self, 'engine', None)
        taker = getattr(engine, 'taker_idx', None)
        is_attacker = (taker is not None) and (taker % 2 == self.seat % 2)
        # "main taker"/"main attacker" = the initiator of the retained
        # contract (§2.1.1.1), not necessarily engine.taker_idx (author of
        # the last bid) -- see _attaquant_principal_seat.
        is_taker = (self._attaquant_principal_seat() == self.seat)
        master_hi, master_lo = self._master_ranks(trump)

        if not trick:
            choice = self._lead_card(trump, is_attacker, is_taker, master_hi, master_lo)
        else:
            choice = self._follow_card(trick, trump, is_attacker, master_hi, master_lo)

        self.hand.remove(choice)
        return choice

SEAT_LABELS = {0: 'N', 1: 'W', 2: 'S', 3: 'E'}  # must stay aligned with render_history.PLAYER_POSITIONS
SUIT_SYMBOLS = {'P': '♠', 'C': '♥', 'K': '♦', 'T': '♣'}
_RED_SUITS = {'C', 'K'}
_ANSI_RED = '\033[91m'
_ANSI_RESET = '\033[0m'


def _trump_label(trump: str) -> str:
    return SUIT_SYMBOLS.get(trump, trump)


def _fmt_card(card: Card) -> str:
    text = f"{card.rank}{SUIT_SYMBOLS.get(card.suit, card.suit)}"
    if card.suit in _RED_SUITS:
        return f"{_ANSI_RED}{text}{_ANSI_RESET}"
    return text


def _fmt_hand(hand: List[Card], trump: Optional[str] = None) -> str:
    groups = []
    for s in SUITS:
        cards = [c for c in hand if c.suit == s]
        if not cards:
            continue
        order = TRUMP_ORDER if (trump == 'TA' or s == trump) else NORMAL_ORDER
        cards = sorted(cards, key=lambda c: order.index(c.rank) if c.rank in order else 99)
        groups.append(' '.join(_fmt_card(c) for c in cards))
    return '  |  '.join(groups) if groups else '(empty)'


def _fmt_bid(bid) -> str:
    level, trump, coinched, capot = bid
    label = f"capot{_trump_label(trump)}" if capot else f"{level}{_trump_label(trump)}"
    if coinched:
        label += " (coinched)"
    return label


def _parse_bid_str(raw: str):
    s = raw.strip().upper().replace(' ', '')
    if not s:
        return None
    valid_trumps = SUITS + ['SA', 'TA']
    if s.startswith('CAPOT'):
        trump = s[len('CAPOT'):]
        return (250, trump, False, True) if trump in valid_trumps else None
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i == 0 or s[i:] not in valid_trumps:
        return None
    return (int(s[:i]), s[i:], False, False)


def _find_card(raw: str, hand: List[Card]) -> Optional[Card]:
    if len(raw) < 2:
        return None
    suit, rank = raw[-1], raw[:-1]
    for c in hand:
        if c.suit == suit and c.rank == rank:
            return c
    return None


def _card_from_repr(raw: str) -> Card:
    return Card(raw[-1], raw[:-1])


class HumanPlayer(Player):
    """Keyboard-controlled player: bid()/play_card() display the current state
    (read back from self.engine.history, like HeuristicPlayer) and read the
    response from stdin, reusing the same legality checks as the engine."""

    def __init__(self, name: str):
        super().__init__(name)
        self._deal_count = 0
        self._shown_tricks = 0
        self.advisor: Optional[Player] = None

    def deal(self, hand: List[Card]):
        super().deal(hand)
        self._deal_count += 1
        self._shown_tricks = 0
        if self._deal_count > 1:
            print("\n(Everyone passed — new deal.)")
        if self.advisor is not None:
            self.advisor.deal(list(hand))

    def _advisor_hint(self, kind: str, current_best=None, leader=None, trick=None, trump=None):
        """Queries self.advisor (an independent Player, never in engine.players)
        about what it would have chosen in place of the human player, without
        touching the real hand: advisor.hand is resynced to a COPY of
        self.hand on each call (advisor.play_card() mutates it by removing
        the chosen card), advisor.initial_hand stays the one fixed at
        deal() (needed by the discard heuristics based on the starting hand)."""
        if self.advisor is None:
            return
        self.advisor.seat = self.seat
        self.advisor.engine = self.engine
        self.advisor.hand = list(self.hand)
        if kind == 'bid':
            suggestion = self.advisor.bid(current_best)
            label = 'pass' if suggestion is None else _fmt_bid(self.engine._normalize_bid(suggestion))
            verb = 'bid'
        else:
            suggestion = self.advisor.play_card(self.seat, leader, trick, trump)
            label = _fmt_card(suggestion)
            verb = 'played'
        print(f"[AI] It would have {verb}: {label}")

    def _label(self, seat: int) -> str:
        return SEAT_LABELS.get(seat, str(seat))

    def _print_trick_recap(self, trick_record: dict, trick_no: int):
        plays = '  '.join(
            f"{self._label(p['seat'])}:{_fmt_card(_card_from_repr(p['card']))}"
            for p in trick_record['plays']
        )
        winner = trick_record['winner']
        print(f"\n[Trick {trick_no} done] {plays}  →  {self._label(winner)} ({self._relation(winner)}) wins")

    def show_new_tricks(self):
        """Catches up the display of finished tricks since the last call
        (self.engine.history is the only channel for knowing what's been
        played -- needed when this player isn't the last to play in a
        trick: the last 1-2 cards are played "silently" by the bots before
        it's their turn again)."""
        tricks = self.engine.history['tricks']
        while self._shown_tricks < len(tricks):
            self._print_trick_recap(tricks[self._shown_tricks], self._shown_tricks + 1)
            self._shown_tricks += 1

    def _relation(self, other_seat: int) -> str:
        if other_seat == self.seat:
            return 'you'
        if other_seat == (self.seat + 2) % 4:
            return 'your partner'
        return 'opponent'

    def bid(self, current_best):
        engine = self.engine
        auction = engine.history['auction']
        parts = [
            f"{self._label(e['seat'])} pass" if e['offer'] is None
            else f"{self._label(e['seat'])} {_fmt_bid(e['offer'])}"
            for e in auction
        ]
        auction_line = (' · '.join(parts) + ' · → your turn') if parts else '→ your turn'

        print(f"\n=== Bidding — {self._label(self.seat)} (you) ===")
        print(f"Bids: {auction_line}")
        print(f"Your hand: {_fmt_hand(self.hand)}")
        if current_best is not None:
            last = next((e for e in reversed(auction) if e['offer'] is not None), None)
            bidder = self._label(last['seat']) if last else '?'
            who = self._relation(last['seat']) if last else '?'
            print(f"Current contract: {_fmt_bid(current_best)} ({bidder}, {who})")
        else:
            print("Current contract: none")
        self._advisor_hint('bid', current_best=current_best)

        while True:
            choice = input("[p] pass   [b] bid   [c] coinche > ").strip().lower()
            if choice in ('', 'p', 'pass'):
                return None
            if choice in ('c', 'coinche', 'double'):
                if current_best is None:
                    print("  Can't coinche: no bid in progress.")
                    continue
                return (current_best[0], current_best[1], True, current_best[3])
            if choice in ('b', 'bid', 'raise'):
                raw = input("  Your bid (e.g. 90C, 100TA, capotP): ").strip()
                parsed = _parse_bid_str(raw)
                if parsed is None:
                    print("  Unrecognized format.")
                    continue
                normalized = engine._normalize_bid(parsed)
                if not engine._is_valid_bid(normalized, current_best):
                    print("  Invalid bid (level or suit not allowed).")
                    continue
                return normalized
            print("  Unrecognized choice.")

    def play_card(self, seat: int, leader: int, trick: list, trump: str):
        engine = self.engine
        self.show_new_tricks()
        trick_no = len(engine.history['tricks']) + 1
        contract = engine.contract
        contract_label = f"capot{_trump_label(trump)}" if contract.capot else f"{contract.level}{_trump_label(trump)}"
        print(f"\n=== Trick {trick_no}/8 — {self._label(seat)} (you) ===")
        print(f"Contract: {contract_label} by {self._label(engine.taker_idx)} ({self._relation(engine.taker_idx)})")

        played = dict(trick)
        for s in range(4):
            tag = '(you)' if s == seat else ('(partner)' if s == (seat + 2) % 4 else '(opp.)')
            if s in played:
                status = f"[ {_fmt_card(played[s])} ]"
                if s == leader:
                    status += " [lead]"
            elif s == seat:
                status = "(your turn)"
            else:
                status = "(waiting)"
            print(f"  {self._label(s):1s} {tag:12s} {status}")

        print(f"Your hand: {_fmt_hand(self.hand, trump)}")
        self._advisor_hint('play', leader=leader, trick=trick, trump=trump)

        legal = engine.legal_moves(seat, self.hand, trick, trump)
        while True:
            raw = input("Your card (code, e.g. 8K): ").strip().upper()
            card = _find_card(raw, self.hand)
            if card is None:
                print("  Invalid card or not in your hand.")
                continue
            if card not in legal:
                print("  Illegal move (you must follow suit/cut/overtrump per the rules).")
                continue
            self.hand.remove(card)
            return card


class RLPlayer(HeuristicPlayer):
    """Bids exactly like HeuristicPlayer (bid() inherited as-is); only card
    play is delegated to a trainable policy. With no policy attached, falls
    back to heuristic play (a fallback, not a buggy random player)."""
    def __init__(self, name:str, policy=None):
        super().__init__(name)
        self.policy = policy

    def play_card(self, seat:int, leader:int, trick:list, trump:str):
        if self.policy is None:
            return super().play_card(seat, leader, trick, trump)
        legal = self.engine.legal_moves(seat, self.hand, trick, trump)
        card = self.policy.choose_card(self, legal, leader, trick, trump)
        self.hand.remove(card)
        return card


def create_player(strategy: str, name: str):
    s = strategy.lower()
    if s == 'random':
        return RandomPlayer(name)
    if s == 'human':
        return HumanPlayer(name)
    if s.startswith('heuristic') or s == 'user':
        # allow variants like 'heuristic:user' or 'heuristic:assistant'
        parts = s.split(':')
        variant = 'user' if len(parts) == 1 else parts[1]
        return HeuristicPlayer(name, variant=variant)
    if s == 'rl' or s == 'agent':
        return RLPlayer(name)
    # default
    return RandomPlayer(name)


#For manual testing

def main():
    print("ok")  # sanity check
    hand = [Card('P','A'), Card('P','10'), Card('K','A'), Card('T','Q'), Card('T','J'), Card('T','9'), Card('T','8'), Card('T','7')]
    player1=HeuristicPlayer("Player1")
    player1.deal(hand)
    print("Player1 hand:", hand)
    rep = player1.bid(None)
    print("Player1 bid:", rep)
    print(hand[1].suit)

if __name__ == '__main__':
    main()
