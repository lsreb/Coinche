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

    @staticmethod
    def _normalize_bid(bid):
        level, trump, coinched, capot = bid
        if capot or level == 250:
            level = 250
            capot = True
        return (level, trump, coinched, capot)

    @staticmethod
    def _is_valid_bid(bid, current_best):
        if bid is None:
            return False
        level, trump, coinched, capot = bid
        if capot or level == 250:
            level = 250
        if level < 80 or level > 250 or level % 10 != 0:
            return False
        if current_best is None:
            # Coincher suppose une annonce existante a doubler (regles_coinche.md).
            return not coinched
        if coinched:
            # La coinche double la derniere annonce telle quelle (meme palier/couleur/
            # capot), elle ne surencherit pas de 10 comme une enchere normale.
            best_level, best_trump, _, best_capot = current_best
            return level == best_level and trump == best_trump and capot == best_capot
        return level >= current_best[0] + 10

    def run_auction(self):
        passes = 0
        best = None
        coinched = False
        idx = (self.dealer + 1) % 4
        while passes < 4:
            player = self.players[idx]
            current_best = best[1] if best else None
            b = player.bid(current_best)
            if b is not None:
                b = self._normalize_bid(b)
            if b is None or not self._is_valid_bid(b, current_best):
                if self.history is not None:
                    self.history['auction'].append({'seat': idx, 'offer': None})
                passes += 1
            elif b[2]:
                # Coinche : fixe immediatement le contrat a la derniere annonce
                # (regles_coinche.md), sans laisser les joueurs suivants encherir.
                if self.history is not None:
                    self.history['auction'].append({'seat': idx, 'offer': b})
                coinched = True
                break
            else:
                if self.history is not None:
                    self.history['auction'].append({'seat': idx, 'offer': b})
                best = (idx, b)
                passes = 0
            idx = (idx+1)%4

        if best is None:
            # All players passed: redeal
            self.dealer = (self.dealer + 1) % 4
            self.deal()
            self.run_auction()
        else:
            bidder_idx, (level, trump, _, capot) = best
            self.contract = Contract(level, trump, coinched, capot)
            self.taker_idx = bidder_idx

        if self.history is not None:
            self.history['contract'] = {
                'level': self.contract.level,
                'trump': self.contract.trump,
                'taker': getattr(self, 'taker_idx', None),
                'capot': self.contract.capot,
                'coinched': self.contract.coinched,
            }

    def card_order_key(self, card:Card, lead_suit:Optional[str], trump:str):
        if trump == 'TA' or (card.suit == trump):
            order = TRUMP_ORDER
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

    def legal_moves(self, seat: int, hand: List[Card], trick: List[Tuple[int, Card]], trump: str) -> List[Card]:
        """Cartes que `seat` peut légalement jouer dans `trick` en cours, régi par
        regles_coinche.md §4. `GameEngine.play()` ne valide jamais lui-même la
        légalité de ce que `play_card()` retourne : c'est entièrement la
        responsabilité de l'appelant (joueur ou policy)."""
        if not trick:
            return list(hand)

        lead_suit = trick[0][1].suit

        if trump in ('SA', 'TA'):
            same = [c for c in hand if c.suit == lead_suit]
            return same if same else list(hand)

        same = [c for c in hand if c.suit == lead_suit]
        trumps = [c for c in hand if c.suit == trump]
        partner_idx = (seat + 2) % 4

        current_winner = trick[0]
        for t in trick[1:]:
            if self.card_order_key(t[1], lead_suit, trump) > self.card_order_key(current_winner[1], lead_suit, trump):
                current_winner = t

        if lead_suit == trump:
            # Atout demandé : on doit jouer atout (en montant en force si possible),
            # sinon défausse libre si on n'a plus d'atout.
            if not trumps:
                return list(hand)
            beating = [c for c in trumps
                       if self.card_order_key(c, lead_suit, trump) > self.card_order_key(current_winner[1], lead_suit, trump)]
            return beating if beating else list(trumps)

        if same:
            return same
        if not trumps:
            return list(hand)
        if current_winner[0] == partner_idx:
            # Partenaire déjà maître du pli : défausse libre (la coupe reste un choix possible).
            return list(hand)
        if current_winner[1].suit == trump:
            # Un adversaire a déjà coupé : on doit surcouper si possible, sinon défausse libre.
            beating = [c for c in trumps
                       if self.card_order_key(c, lead_suit, trump) > self.card_order_key(current_winner[1], lead_suit, trump)]
            return beating if beating else list(hand)
        # Personne n'a encore coupé : on doit couper (n'importe quel atout).
        return list(trumps)

    def card_point(self, card:Card, trump:str):
        if trump == 'TA' or card.suit == trump:
            return TRUMP_POINTS.get(card.rank, 0)
        elif trump == 'SA':
            return SA_POINTS.get(card.rank, 0)
        else:
            return NORMAL_POINTS.get(card.rank, 0)

    @staticmethod
    def _round_dizaine(points: int, direction: str = 'nearest') -> int:
        """Arrondit `points` a la dizaine (regles_coinche.md §5). 'nearest' :
        unites 5-9 vers la dizaine superieure, 1-4 vers l'inferieure (arrondi
        standard "half up", pas le round-half-to-even de Python) -- utilise
        pour les contrats couleur/SA, ou attaque et defense arrondissent
        pareil. 'down'/'up' forcent systematiquement vers le bas/le haut --
        utilise a TA, ou l'arrondi est explicitement asymetrique (attaque
        vers le bas, defense vers le haut)."""
        if direction == 'down':
            return (points // 10) * 10
        if direction == 'up':
            return -(-points // 10) * 10
        return ((points + 5) // 10) * 10

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

        # scoring -- regles_coinche.md §5
        team_tricks = {0: 0, 1: 0}
        for p, cards in tricks_won.items():
            team_tricks[p % 2] += len(cards) // 4
        capot_team = 0 if team_tricks[0] == 8 else (1 if team_tricks[1] == 8 else None)

        raw_card_points = {0: 0, 1: 0}
        for p, cards in tricks_won.items():
            raw_card_points[p % 2] += sum(self.card_point(c, trump) for c in cards)
        if trump != 'TA':
            raw_card_points[leader % 2] += 10  # 10 de der (pas de der a TA)

        belote_bonus = {0:0,1:0}
        if trump != 'SA' and trump != 'TA':
            for p, player in enumerate(self.players):
                initial_ranks = {(c.suit,c.rank) for c in player.initial_hand}
                if (trump,'K') in initial_ranks and (trump,'Q') in initial_ranks:
                    belote_bonus[p%2] += 20
                    break

        # Points de plis bruts (+ der + belote), toujours calcules independamment
        # de la coinche/reussite du contrat : c'est ce qui determine la reussite
        # du contrat (regles_coinche.md : "avec la belote elle passe a 95 et
        # pourrait faire son contrat a 80-C", la belote compte donc bien pour la
        # reussite), et ce que des scripts externes (simulate_contracts.py)
        # veulent pour calibrer les encheres independamment du score final.
        raw_points = {team: raw_card_points[team] + belote_bonus[team] for team in (0, 1)}
        if self.history is not None:
            history_raw_points = dict(raw_points)
            if capot_team is not None:
                # Un capot vaut "officiellement" 250 (regles_coinche.md), y compris
                # pour ce champ d'historique -- simulate_contracts.py s'appuie sur
                # raw_points atteignant 250 pour detecter un capot reussi dans son
                # tableau de calibration. N'affecte pas la logique de scoring
                # ci-dessous (capot_team est deja traite dans une branche a part).
                history_raw_points[capot_team] = 250 + belote_bonus[capot_team]
            self.history['raw_points'] = history_raw_points

        taker_team = self.taker_idx % 2
        defense_team = 1 - taker_team
        # A TA le seuil de reussite est sur l'echelle a 15 (80-TA necessite 120,
        # 90-TA necessite 135, ...), pas la valeur brute du contrat (correspondance
        # 80-90-...-160 <-> 120-135-...-240, regles_coinche.md §5).
        threshold = int(self.contract.level * 1.5) if trump == 'TA' else self.contract.level

        if capot_team is not None:
            # Une equipe qui fait tous les plis marque 250 (quel que soit le
            # contrat), 500 (+belote) en plus si c'etait precisement le capot
            # annonce et reussi par le preneur (regles_coinche.md §5).
            attack_success = (capot_team == taker_team)
            if self.contract.capot and attack_success:
                team_points = {taker_team: 500 + belote_bonus[taker_team], defense_team: belote_bonus[defense_team]}
            else:
                team_points = {capot_team: 250, (1 - capot_team): 0}
        elif self.contract.capot:
            # Capot annonce mais pas realise (preneur n'a pas fait les 8 plis) :
            # chute du contrat capot -- self.contract.level vaut deja 250 pour un
            # capot (_normalize_bid), donc la defense marque 160+250.
            attack_success = False
            team_points = {
                taker_team: belote_bonus[taker_team],
                defense_team: 160 + self.contract.level + belote_bonus[defense_team],
            }
        else:
            attack_success = raw_points[taker_team] >= threshold
            if attack_success:
                if trump == 'TA':
                    # A TA, les points de plis sont d'abord convertis sur
                    # l'echelle a 15 (15 points TA = 10 points "equivalents"),
                    # PUIS arrondis a la dizaine -- pas l'inverse. En pratique
                    # ca revient a une division entiere par 15 (attaque, vers
                    # le bas) ou une division entiere superieure par 15
                    # (defense, vers le haut), le tout x10. Le contrat ajoute
                    # est le seuil deja mis a l'echelle (`threshold`, ex. 120
                    # pour un contrat annonce a 80), pas la valeur brute
                    # annoncee -- confirme par regles_coinche.md §5 (exemple
                    # corrige : 195-209 points de plis -> 130 points
                    # equivalents pour l'attaque, 120+130=250 ; le complementaire
                    # cote defense donne 30).
                    attack_made = (raw_card_points[taker_team] // 15) * 10
                    defense_made = -(-raw_card_points[defense_team] // 15) * 10
                    contract_component = threshold
                else:
                    # Couleur/SA : arrondi standard (5-9 superieur, 1-4
                    # inferieur) des deux cotes, contrat ajoute a sa valeur
                    # annoncee brute (regles_coinche.md §5, exemple 90-T
                    # verifie : 107->110, 55->60).
                    attack_made = self._round_dizaine(raw_card_points[taker_team], 'nearest')
                    defense_made = self._round_dizaine(raw_card_points[defense_team], 'nearest')
                    contract_component = self.contract.level
                team_points = {
                    taker_team: contract_component + attack_made + belote_bonus[taker_team],
                    defense_team: defense_made + belote_bonus[defense_team],
                }
            else:
                # Contrat chute : l'attaque marque 0 (+sa belote), la defense
                # marque 160 + la valeur du contrat (+sa belote).
                team_points = {
                    taker_team: belote_bonus[taker_team],
                    defense_team: 160 + self.contract.level + belote_bonus[defense_team],
                }

        if self.contract.coinched:
            # La coinche double la mise et remplace le score par un forfait fixe
            # (160 + 2x le contrat) pour l'equipe qui gagne la donne (= a reussi
            # son contrat, deja determine ci-dessus via `attack_success`), 0 pour
            # l'autre -- independamment des points de plis reellement faits.
            winner = taker_team if attack_success else defense_team
            loser = 1 - winner
            team_points = {
                winner: 160 + 2 * self.contract.level + belote_bonus[winner],
                loser: belote_bonus[loser],
            }

        return team_points, self.contract
    

if __name__ == '__main__':
    print("ok") #le test
    sample_hands = [['AT', '10T', 'KC', 'QC', 'JP', '9P', '8P', '7P'],
            ['AK', '10K', 'KT', 'QT', 'JC', '9C', '8C', '7C'],
            ['AP', '10P', 'KK', 'QK', 'JT', '9T', '8T', '7T'],
            ['AC', '10C', 'KP', 'QP', 'JK', '9K', '8K', '7K']]