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
    # Enchères (section 1 de heuristiques.md)
    # ---------------------------------------------------------------

    def _last_bid_info(self):
        """Retourne (seat, bid) de la dernière enchère réellement faite dans l'enchère en cours."""
        engine = getattr(self, 'engine', None)
        if engine is None or not getattr(engine, 'history', None):
            return None
        for entry in reversed(engine.history.get('auction', [])):
            if entry.get('offer') is not None:
                return entry['seat'], entry['offer']
        return None

    def _count_tricks(self, trump_suit: str) -> int:
        # 1 pli par atout, puis en dehors des atouts : 1/As, 1/10 troisième,
        # 2 pour 10+K troisième, 2 pour As+10 même couleur.
        tricks = count_suit(self.hand, trump_suit)
        for s in SUITS:
            if s == trump_suit:
                continue
            cards = [c for c in self.hand if c.suit == s]
            ranks = {c.rank for c in cards}
            if 'A' in ranks and '10' in ranks:
                tricks += 2
            elif '10' in ranks and 'K' in ranks and len(cards) >= 3:
                tricks += 2
            elif 'A' in ranks:
                tricks += 1
            elif '10' in ranks and len(cards) >= 3:
                tricks += 1
        return tricks

    def _decrochage_ok(self, own_level: int, opp_level: int) -> bool:
        # Le décrochage tolère un "petit mensonge" d'un seul palier (10 points) au-delà
        # de ce que la main justifie vraiment : une main à 100 peut dire 110 quand
        # l'adversaire a dit 100, mais une main à 80 ne peut pas sauter jusqu'à 110.
        return own_level >= opp_level - 10

    def _my_last_bid(self):
        """Retourne ma propre dernière offre faite dans l'enchère en cours (ou None)."""
        engine = getattr(self, 'engine', None)
        seat = getattr(self, 'seat', None)
        if engine is None or not getattr(engine, 'history', None):
            return None
        for entry in reversed(engine.history.get('auction', [])):
            if entry['seat'] == seat and entry.get('offer') is not None:
                return entry['offer']
        return None

    def _sa_known_aces(self, bidder_seat, bid):
        # Comme pour un décrochage à la couleur : si l'annonce SA de `bidder_seat` suit
        # immédiatement une offre adverse d'un palier inférieur, elle peut elle-même être
        # un décrochage (donc représenter un as de moins que ce que le palier indique).
        # On retrouve l'enchère précise dans l'historique (pas juste l'avant-dernière),
        # car cette fonction peut être appelée bien après coup, sur une enchère passée.
        level = bid[0]
        aces_table = {80: 2, 90: 3, 100: 4}
        engine = getattr(self, 'engine', None)
        if engine is not None and getattr(engine, 'history', None):
            made = [(e['seat'], tuple(e['offer'])) for e in engine.history.get('auction', []) if e.get('offer') is not None]
            for i, (seat, offer) in enumerate(made):
                if seat == bidder_seat and offer == tuple(bid):
                    if i >= 1:
                        prev_seat, prev_offer = made[i - 1]
                        if prev_seat % 2 != bidder_seat % 2 and prev_offer[0] == level - 10:
                            return aces_table.get(level - 10, 0)
                    break
        return aces_table.get(level, 0)

    def _partner_sa_to_color(self, partner_seat, partner_bid, own_color_candidates):
        # On peut basculer sur sa propre couleur plutôt que de simplement soutenir le
        # SA du partenaire, en s'appuyant sur les as déjà révélés par son annonce SA
        # (moins un, potentiellement à l'atout choisi) : voir heuristiques.md §1.1.
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
        # Chaque type d'information (soutien d'atout, as exter) n'est révélé qu'une
        # fois par donne : deux partenaires ne doivent pas se relancer indéfiniment
        # sur la même carte, mais peuvent remonter à des tours différents pour des
        # raisons différentes.
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
                # Mon partenaire a basculé sur cette couleur (_partner_sa_to_color) en se
                # basant sur mon SA : il a déjà crédité (as supposés - 1) de mes as. Je ne
                # dois ajouter que ceux qui restent, pas recompter mes as moins un.
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

    def _partner_sa_remonte(self, partner_bid):
        level = partner_bid[0]
        contrib = self._raise_contributions
        my_aces = sum(1 for c in self.hand if c.rank == 'A')
        new_level = level
        if my_aces > 0 and 'sa_aces' not in contrib:
            new_level += 10 * my_aces
            contrib.add('sa_aces')
        min_partner_aces = {80: 2, 90: 3, 100: 4}.get(level, 0)
        if min_partner_aces + my_aces >= 4 and 'sa_tens' not in contrib:
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
        min_partner_jacks = {80: 2, 90: 3, 100: 4}.get(level, 0)
        if min_partner_jacks + my_jacks >= 4 and 'ta_nines' not in contrib:
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

        candidates = []

        # 1.1) Couleur
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

        # 1.3) TA (Valets à la place des As, 9 à la place des 10)
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

        # Remontée du partenaire
        if partner_bid is not None:
            if partner_bid[1] == 'SA':
                r = self._partner_sa_remonte(partner_bid)
                if r is not None:
                    candidates.append(r)
                own_color_candidates = [c for c in candidates if c[1] in SUITS and c is not r]
                r2 = self._partner_sa_to_color(last[0], partner_bid, own_color_candidates)
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

        # Enchère maximale entre toutes les couleurs/types disponibles
        best_offer = max(candidates, key=lambda b: b[0]) if candidates else None

        # Nos propres As/Valets sont déjà "annoncés" dès notre déclaration initiale (le
        # niveau 80/90/100 les encode) : on ne doit pas pouvoir les re-compter plus tard
        # comme une remontée, sous peine de gonfler artificiellement le contrat.
        if best_offer is not None and best_offer == sa_candidate:
            self._raise_contributions.add('sa_aces')
        if best_offer is not None and best_offer == ta_candidate:
            self._raise_contributions.add('ta_jacks')

        # Décrochage : uniquement en réponse à un adversaire (jamais au partenaire), et
        # seulement si la main n'est pas à plus d'un palier de l'adversaire.
        if opponent_bid is not None and best_offer is not None:
            opp_level = opponent_bid[0]
            if opp_level <= 100 and best_offer[0] <= opp_level:
                next_level = opp_level + 10
                trump = best_offer[1]
                can_decrochage = next_level <= 110 and self._decrochage_ok(best_offer[0], opp_level)
                if can_decrochage and next_level >= 100 and trump in SUITS:
                    # Sans certitude que le partenaire ait le 9, un décrochage jusqu'à 100
                    # n'est crédible que si on compte au moins 4 plis dans sa propre main
                    # (l'as/dix exter éventuel est déjà inclus dans ce décompte).
                    can_decrochage = self._count_tricks(trump) >= 4
                if can_decrochage:
                    best_offer = (next_level, trump, best_offer[2], best_offer[3])
                    if next_level >= 100 and trump in SUITS:
                        # Le soutien d'atout et l'as exter éventuel ont déjà servi à
                        # justifier ce décrochage : on ne doit pas pouvoir les re-annoncer
                        # comme une remontée supplémentaire quand le partenaire relance.
                        self._raise_contributions.add('color_support')
                        self._raise_contributions.add('color_exter')

        return best_offer

    # ---------------------------------------------------------------
    # Jeu de la carte (section 2 de heuristiques.md)
    # ---------------------------------------------------------------

    def _rank_strength(self, card: Card, trump: str, lead_suit: Optional[str]):
        # (is_trump, is_lead, -rank_idx) : même logique que GameEngine.card_order_key,
        # pour qu'un atout batte toujours une carte hors-atout lors des comparaisons entre couleurs.
        is_trump = 1 if (trump == 'TA' or card.suit == trump) else 0
        order = TRUMP_ORDER if is_trump else NORMAL_ORDER
        try:
            rank_idx = order.index(card.rank)
        except ValueError:
            rank_idx = 99
        is_lead = 1 if (lead_suit is not None and card.suit == lead_suit) else 0
        return (is_trump, is_lead, -rank_idx)

    def _master_ranks(self, trump: str):
        # Sous TA, toutes les couleurs se comptent comme à l'atout : Valet puis 9.
        return ('J', '9') if trump == 'TA' else ('A', '10')

    def _card_seen_before(self, suit: str, rank: str) -> bool:
        engine = getattr(self, 'engine', None)
        if engine is None or not getattr(engine, 'history', None):
            return False
        target = f"{rank}{suit}"
        for t in engine.history.get('tricks', []):
            for p in t['plays']:
                if p['card'] == target:
                    return True
        return False

    def _trumps_remain_with_opponents(self, trump: str) -> bool:
        engine = getattr(self, 'engine', None)
        seen = sum(1 for c in self.hand if c.suit == trump)
        if engine is not None and getattr(engine, 'history', None):
            for t in engine.history.get('tricks', []):
                for p in t['plays']:
                    if p['card'][-1] == trump:
                        seen += 1
        return seen < 8

    def _trump_already_led(self, trump: str) -> bool:
        # La retenue du "9 second" ne vaut que pour le tout premier tour d'atout :
        # au tour suivant, la consigne s'inverse (on joue le 9 si on l'a).
        engine = getattr(self, 'engine', None)
        if engine is None or not getattr(engine, 'history', None):
            return False
        for t in engine.history.get('tricks', []):
            for p in t['plays']:
                if p['card'][-1] == trump:
                    return True
        return False

    def _is_confirmed_master(self, card: Card, trump: str) -> bool:
        # Vraie même si ce n'est ni l'as ni le 10 : toutes les cartes qui la
        # dominent dans sa couleur sont déjà tombées, donc elle gagne à coup sûr.
        order = TRUMP_ORDER if (trump == 'TA' or card.suit == trump) else NORMAL_ORDER
        higher_ranks = order[:order.index(card.rank)]
        return all(self._card_seen_before(card.suit, r) for r in higher_ranks)

    def _choose_attack_suit(self, trump):
        # L'attaque veut ouvrir dans la couleur où elle avait initialement le plus
        # de cartes (heuristiques.md §2.1.1.2), pas juste jouer la carte la plus
        # faible toutes couleurs confondues.
        initial = getattr(self, 'initial_hand', self.hand)
        lengths = {s: sum(1 for c in initial if c.suit == s) for s in SUITS if s != trump}
        available = [s for s in lengths if count_suit(self.hand, s) > 0]
        if not available:
            return None
        return max(available, key=lambda s: lengths[s])

    def _lead_offsuit_master(self, trump, master_hi, master_lo):
        offsuit = [c for c in self.hand if c.suit != trump]
        if not offsuit:
            return min(self.hand, key=lambda c: self._rank_strength(c, trump, None))
        masters = [c for c in offsuit if c.rank == master_hi]
        if masters:
            return masters[0]
        seconds = [c for c in offsuit if c.rank == master_lo and self._card_seen_before(c.suit, master_hi)]
        if seconds:
            return seconds[0]
        confirmed = [c for c in offsuit if self._is_confirmed_master(c, trump)]
        if confirmed:
            return max(confirmed, key=lambda c: self._rank_strength(c, trump, None))
        suit = self._choose_attack_suit(trump)
        pool = [c for c in offsuit if c.suit == suit] if suit else offsuit
        return min(pool, key=lambda c: self._rank_strength(c, trump, None))

    def _lead_taker_trump(self, trump, master_hi, master_lo):
        if has_rank(self.hand, trump, 'J'):
            return next(c for c in self.hand if c.suit == trump and c.rank == 'J')
        trumps = [c for c in self.hand if c.suit == trump]
        if trumps:
            if self._trump_already_led(trump) and not self._trumps_remain_with_opponents(trump):
                # plus aucun atout ne reste chez l'adversaire : inutile de jouer notre
                # dernier atout (on le sait seul à en avoir), autant le garder pour une
                # coupe éventuelle et attaquer ailleurs.
                return self._lead_offsuit_master(trump, master_hi, master_lo)
            # 9 second : uniquement au tout premier tour d'atout, on garde le 9
            # (notre plus gros atout, faute de valet) pour le tour suivant.
            if (not self._trump_already_led(trump)) and has_rank(self.hand, trump, '9') and len(trumps) > 1:
                others = [c for c in trumps if c.rank != '9']
                return min(others, key=lambda c: self._rank_strength(c, trump, None))
            if has_rank(self.hand, trump, '9'):
                return next(c for c in trumps if c.rank == '9')
            if self._trumps_remain_with_opponents(trump):
                return min(trumps, key=lambda c: self._rank_strength(c, trump, None))
        return self._lead_offsuit_master(trump, master_hi, master_lo)

    def _lead_partner_trump(self, trump, master_hi, master_lo):
        trumps = [c for c in self.hand if c.suit == trump]
        if trumps:
            if self._trump_already_led(trump) and not self._trumps_remain_with_opponents(trump):
                # même raison que côté attaquant principal : ne pas gâcher le dernier
                # atout quand on sait qu'on est seul à en tenir encore.
                return self._lead_offsuit_master(trump, master_hi, master_lo)
            best = max(trumps, key=lambda c: self._rank_strength(c, trump, None))
            if best.rank == '9' and len(trumps) > 1 and not self._trump_already_led(trump):
                # même logique que le tour du valet : on garde le 9 pour le tour suivant
                others = [c for c in trumps if c.rank != '9']
                return min(others, key=lambda c: self._rank_strength(c, trump, None))
            return best
        return self._lead_offsuit_master(trump, master_hi, master_lo)

    def _lead_defense(self, trump, master_hi, master_lo):
        offsuit_masters = [c for c in self.hand if c.rank == master_hi and c.suit != trump]
        if offsuit_masters:
            return offsuit_masters[0]
        singletons = [s for s in SUITS if s != trump and count_suit(self.hand, s) == 1]
        if singletons:
            return next(c for c in self.hand if c.suit == singletons[0])
        return min(self.hand, key=lambda c: self._rank_strength(c, trump, None))

    def _lead_attack_sa_ta(self, trump, master_hi, master_lo):
        long_with_master, long_without_master, short_suits = [], [], []
        for s in SUITS:
            cards = [c for c in self.hand if c.suit == s]
            if not cards:
                continue
            if len(cards) >= 3:
                (long_with_master if any(c.rank == master_hi for c in cards) else long_without_master).append((s, cards))
            else:
                short_suits.append((s, cards))

        if long_with_master:
            s, cards = long_with_master[0]
            if self._card_seen_before(s, master_hi) and any(c.rank == master_lo for c in cards):
                return next(c for c in cards if c.rank == master_lo)
            if len(long_with_master) == 1 and not long_without_master and not short_suits:
                return next(c for c in cards if c.rank == master_hi)
            # on évite de jouer sa carte maîtresse au premier tour d'une couleur si une autre ouverture existe
            if long_without_master:
                _, cards2 = long_without_master[0]
                return min(cards2, key=lambda c: self._rank_strength(c, trump, None))
            if short_suits:
                _, cards2 = short_suits[0]
                return max(cards2, key=lambda c: self._rank_strength(c, trump, None))
            return next(c for c in cards if c.rank == master_hi)

        if long_without_master:
            s, cards = long_without_master[0]
            return min(cards, key=lambda c: self._rank_strength(c, trump, None))

        if short_suits:
            s, cards = short_suits[0]
            return max(cards, key=lambda c: self._rank_strength(c, trump, None))

        return min(self.hand, key=lambda c: self._rank_strength(c, trump, None))

    def _lead_card(self, trump, is_attacker, is_taker, master_hi, master_lo):
        if trump in SUITS:
            if is_attacker and is_taker:
                return self._lead_taker_trump(trump, master_hi, master_lo)
            if is_attacker:
                return self._lead_partner_trump(trump, master_hi, master_lo)
            return self._lead_defense(trump, master_hi, master_lo)
        if is_attacker:
            return self._lead_attack_sa_ta(trump, master_hi, master_lo)
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
            # à contrat couleur, seul le Valet d'atout est réellement imparable (une carte
            # maîtresse hors-atout peut toujours être coupée par un adversaire encore pourvu d'atout)
            return card.suit == trump and card.rank == master_rank
        return card.rank == master_rank

    def _choose_defausse_suit(self):
        # couleur faible = singlette initiale, ou couleur à 2 cartes initiales sans 10 ni As
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
            return max(pool, key=lambda c: self._rank_strength(c, trump, None))
        return min(cand, key=lambda c: self._rank_strength(c, trump, None))

    def _secure_when_partner_wins(self, same, trump, lead, trick):
        # Le pli est déjà gagné pour mon camp : je choisis laquelle de mes cartes de
        # cette couleur jouer maintenant (sans risque), et laquelle garder en main.
        order = TRUMP_ORDER if (trump == 'TA' or lead == trump) else NORMAL_ORDER
        ordered = sorted(same, key=lambda c: self._rank_strength(c, trump, lead), reverse=True)
        top = ordered[0]
        higher_ranks = order[:order.index(top.rank)]
        top_secured = all(
            self._card_seen_before(top.suit, r) or any(c.suit == top.suit and c.rank == r for _, c in trick)
            for r in higher_ranks
        )
        if top_secured or len(ordered) == 1:
            # ma meilleure carte sera maitresse plus tard : je la garde
            return ordered[-1]
        if len(ordered) == 2:
            # valet/roi second : je le sécurise maintenant plutôt que la petite carte
            return ordered[0]
        # 3 cartes ou plus : je sécurise la carte intermédiaire, je garde la meilleure en réserve
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
                # Suite du "faire tomber les atouts" (2.1.1.1) : quel que soit le meneur du
                # pli, l'attaque continue la séquence valet puis 9 puis le plus faible.
                for rank in ('J', '9'):
                    for c in same:
                        if c.rank == rank:
                            return c
                return min(same, key=lambda c: self._rank_strength(c, trump, lead))
            winning = [c for c in same if self._rank_strength(c, trump, lead) > self._rank_strength(current_winner[1], trump, lead)]
            if winning:
                # Priorité à la carte maitre (l'As, ou le 10 si l'as est déjà passé) pour
                # devenir maitre du pli, sinon la plus petite carte qui remporte le pli.
                masters = [c for c in winning if c.rank == master_hi]
                if masters:
                    return masters[0]
                seconds = [c for c in winning if c.rank == master_lo and self._card_seen_before(lead, master_hi)]
                if seconds:
                    return seconds[0]
                return min(winning, key=lambda c: self._rank_strength(c, trump, lead))
            partner_idx = (self.seat + 2) % 4
            if current_winner[0] == partner_idx and (
                self._partner_is_absolute_master(current_winner, trump) or len(trick) == 3
            ):
                return self._secure_when_partner_wins(same, trump, lead, trick)
            return min(same, key=lambda c: self._rank_strength(c, trump, lead))

        trumps = [c for c in self.hand if c.suit == trump] if trump in SUITS else []
        if trumps:
            partner_idx = (self.seat + 2) % 4
            if current_winner[0] == partner_idx:
                return min(self.hand, key=lambda c: self._rank_strength(c, trump, None))
            higher_trumps = [c for c in trumps if self._rank_strength(c, trump, lead) >
                              self._rank_strength(current_winner[1], trump, lead)]
            if higher_trumps:
                if is_attacker:
                    # attaque : économie, la plus petite carte qui gagne
                    return min(higher_trumps, key=lambda c: self._rank_strength(c, trump, None))
                # défense : coupe avec le plus gros atout, sauf 9 troisième ou As quatrième
                nine_third = has_rank(self.hand, trump, '9') and len(trumps) == 3
                as_fourth = has_rank(self.hand, trump, 'A') and len(trumps) == 4
                if nine_third or as_fourth:
                    return min(higher_trumps, key=lambda c: self._rank_strength(c, trump, None))
                return max(higher_trumps, key=lambda c: self._rank_strength(c, trump, None))
            return min(self.hand, key=lambda c: self._rank_strength(c, trump, None))

        return self._discard(trump, current_winner)

    def play_card(self, seat:int, leader:int, trick:list, trump:str):
        engine = getattr(self, 'engine', None)
        taker = getattr(engine, 'taker_idx', None)
        is_attacker = (taker is not None) and (taker % 2 == self.seat % 2)
        is_taker = (taker == self.seat)
        master_hi, master_lo = self._master_ranks(trump)

        if not trick:
            choice = self._lead_card(trump, is_attacker, is_taker, master_hi, master_lo)
        else:
            choice = self._follow_card(trick, trump, is_attacker, master_hi, master_lo)

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


#Pour les tests 

def main():
    print("ok") #le test
    hand = [Card('P','A'), Card('P','10'), Card('K','A'), Card('T','Q'), Card('T','J'), Card('T','9'), Card('T','8'), Card('T','7')]
    player1=HeuristicPlayer("Player1")
    player1.deal(hand)
    print("Player1 hand:", hand)
    rep = player1.bid(None)
    print("Player1 bid:", rep)
    print(hand[1].suit)

if __name__ == '__main__':
    main()
