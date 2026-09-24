This file records the rules of belote coinchée, known as coinche.

*(English translation of `regles_coinche.md`, which remains the canonical
source referenced by the code and by `CLAUDE.md`/`heuristiques.md`. If the
rules ever change, update `regles_coinche.md` first and re-sync this file.)*

# Technical Specification: Belote Coinchée for Reinforcement Learning (RL)

## 1) Context

Coinche is a trick-taking card game played with the 32 cards of the 4
suits: the ace (A), king (K), queen (Q), jack (J), 10, 9, 8, and 7. It's
played by 4 players (North, West, South, and East), in two teams of two,
North with South and East with West. It's a trick game, so there are 8
tricks. At the end of a trick, one player wins it, and leads the next one.
Once all 8 tricks are done, the total points from each trick are counted
and compared against the contract to determine who won it. The cards in
hand are hidden from the other players.

A game of coinche is made up of several donnes (deals). A donne goes as
follows: the dealer cuts the deck, deals 8 cards to each player (3 to
each, 3 to each, then 2 to each). Then, starting from the dealer's right,
the players bid. Once all 4 players have passed, bidding ends and the
contract is set. The donne is then played out (the 8 tricks), and the
contract's winner is determined. For the next donne, the dealer becomes
the player to the previous dealer's right.

## 2) Precise sequence of the bidding phase

The player to the dealer's right starts the bidding phase. They announce
a multiple of ten between 80 and 160 inclusive, or capot, or they pass.
They also announce a suit — spades (P), hearts (C), diamonds (K), or
clubs (T) — or "no trump" (SA, sans-atout) or "all trump" (TA,
tout-atout). For example, they might say 80C. The player to their right
must pass or outbid — meaning at least 10 more — in any suit, SA, or TA,
or coinche instead. When a player coinches, the contract is set at the
last bid made, and the stakes are doubled. When three players pass in a
row, the fourth may outbid themselves, but only by changing suit, or to
TA or SA. When all 4 players pass, the contract is set. The team that
took the contract is called the "attacking team", and the other the
"defending team".

If no player made a bid, the donne is void, nobody scores any points, and
play moves to the next donne with the cards reshuffled and a new dealer,
the next player counter-clockwise.

## 3) Card order

In trump: J-9-A-10-K-Q-8-7, with respective point values 20-14-11-10-4-3-0-0.

In normal (non-trump) suits: A-10-K-Q-J-9-8-7, with respective point
values 11-10-4-3-2-0-0-0.

In no-trump (SA), the order is the same as normal suits, but the
respective point values are: 19-10-4-3-2-0-0-0.

In all-trump (TA), the order and values of every suit are those of trump.

When playing a suit contract, trump is stronger than the other three suits.

When playing a suit contract, if any player, attack or defense, holds the
queen and king of trump, they verbally announce "belote" when playing
either one. They then score 20 extra points at scoring time, and these
points can count toward making the contract on attack.

## 4) Play and trick sequence

The player to the dealer's right starts (leads) the first trick with a
card from their hand, which sets the suit led for the trick.

When a player leads a trick, playing counter-clockwise (N > W > S > E >
N), the other players must, if possible, play a card of the same suit as
the first player.

Let's first look at the case of a plain-suit contract. For example, if
spades is trump and the first player leads a heart, everyone must follow
with a heart. A player who can't follow hearts must, in principle, cut —
that is, play trump — if they have any. There are several special cases
here: if their partner is master of the trick, meaning they'd currently
win it, the player may discard instead of cutting, i.e. play any card. If
an opponent has already cut, and the player holds trump, they must
overtrump (cut with a higher card), and if they only hold lower trump,
they may keep it and discard instead. In every other case, a player with
neither the suit led nor any trump must discard.

For example: trump is spades. North leads A-C (ace of hearts), West has
no spades and cuts with 10-P (10 of spades), South has no hearts, and
holds the ace of trump and the 8 of trump, so they must play the ace of
trump, in spades (A-P). East finally has only one trump, the king of
spades K-P, but it doesn't beat the current trump, so they may discard
instead, for example their 7 of clubs (7-T).

Then the player winning the trick, according to the order given in
section 3, leads the next one.

If a player leads trump, the other players must, if possible, play
trump, overtrumping if they can. If they can't, they play any lower
trump. If they have no more trump, they must discard in any suit.

## 5) Scoring

At the end of the 8 tricks, the points from each trick won are counted
separately for attack and defense. If the contract is a suit or SA, the
team that wins the last trick gets a 10-point bonus, called the "10 de
der" (ten for the last trick) or "la der". There's no der in TA. In a
suit or SA, there are therefore 162 points in total. The attack makes its
contract if it scored more points than the number chosen in its
contract. For example, if it scores 75 points from its tricks, including
the der, and its contract was 80-C, then the contract isn't made.

In a suit contract, if the same player plays the king and queen of
trump, they may verbally announce "belote" when playing either one.
Their team then scores a 20-point bonus at the end of the donne, which
can help make the contract. If the team scored 75 points, with the
belote it goes to 95 and could make its 80-C contract, in the previous
example.

To make capot, a team must win all 8 tricks of the donne: capot is a
250-point contract.

If a team makes its contract, and it was an announced capot, it scores
500 plus the 20 points of a possible belote, the defense scores 0. If it
makes any other type of contract, it scores the contract's announced
points plus the points it made, rounding units of 5, 6, 7, 8, and 9 up
to the next ten, and units of 1, 2, 3, and 4 down to the ten below. The
defense scores its own points made, rounded the same way, plus a belote
if it has one. If a team wins every trick, the points made count as 250,
not 160.

For example, for a 90-T contract, the attack made 107, the defense 55
and a belote. The attack scores 90+110, so 200 points, and the defense
60+20, so 80 points.

These points add up from one donne to the next, accumulating, and the
first team to 3000 wins.

If the contract falls short, the attack scores 0 points (plus a possible
20 for its belote), the defense scores 160 + the contract's value, plus
a possible belote.

If a coinche is called, the stakes are doubled. The team that wins the
donne scores 160 + 2 times the contract, the team that loses scores 0.

For scoring in all-trump, there are 248 points in total, no der. To make
80-TA, the attack must score 120 points in TA. Scoring then proceeds in
steps of 15: for 90-TA, the attack must score 135, for 100-TA, 150, and
so on; the mapping is given below:

For 80-90-100-110-120-130-140-150-160 normally, in TA these become:
120-135-150-165-180-195-210-225-240. Capot is still counted as 250.

In TA, the points made banked by the attack or the defense are rounded
down for the attack, and up for the defense: for example, if the attack
makes its 120-TA contract and scores between 195 and 209 points (before
conversion), it scores the 120 of its contract + the 130 equivalent
points earned. The defense scores its 30 defense points.
