# Getting Started — Coinche RL

This document explains, very concretely, what each script in the repo does, how to run them, and how the functions/methods fit together. It's aimed at someone new to the project who hasn't read the code or `CLAUDE.md`.

For a quick reference (canonical commands, architecture summary), see `CLAUDE.md` at the repo root — this document goes further: one entry per function/method, with who calls it and what it calls.

## Table of contents

1. [Repository overview](#1-repository-overview)
2. [Conventions to know before reading the code](#2-conventions-to-know-before-reading-the-code)
3. [Step-by-step walkthrough (commands)](#3-step-by-step-walkthrough-commands)
4. [Detailed reference, file by file](#4-detailed-reference-file-by-file)
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

## 1. Repository overview

The project has two clearly separated layers:

- **The game engine** (`coinche/game.py`, `coinche/players.py`, `coinche/env.py`): implements the rules of Coinche (`regles_coinche.md`) and a hand-written rule-based AI (`HeuristicPlayer`, which follows `heuristiques.md`). Depends on no external library — works even without `torch` installed.
- **The RL layer** (`coinche/rl_agent.py`, `train.py`, `train_ppo.py`, `pretrain.py`, `pretrain_value.py`, `run_growing_pool*.py`, `eval_*.py`): trains a neural network to play cards (not bidding, which always stays handled by the heuristic logic). Requires `torch`; all of this code degrades gracefully (an explicit error or a fallback) if `torch` is missing.

Three "manual" scripts close the loop: `play_test.py` (an isolated game, no human), `play_interactive.py` (a game played on the keyboard against bots), and `render_history.py` (turns a JSON history into a browsable HTML page).

There's no automated test suite in this repo (no `pytest`): verification is done by actually running games and inspecting the produced JSON/HTML, or via the evaluation scripts (`eval_policy.py`/`eval_matchup.py`) which compare checkpoints against each other over a large number of donnes.

### Dependency graph (who imports what)

Notation: `A → B` means "A imports B". A flat list (rather than an ASCII tree) to avoid any ambiguous edge — in particular, `players.py` and `rl_agent.py` have **no** direct dependency on each other, both only depend on `game.py`.

```
coinche/game.py            (no internal dependency)

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

run_growing_pool.py         --(subprocess)--> train.py
run_growing_pool_ppo.py     --(subprocess)--> train_ppo.py

eval_policy.py               → game.py, players.py, rl_agent.py, train_ppo.py (Policy classes)
eval_matchup.py               → players.py, rl_agent.py, train_ppo.py, eval_policy.py
                                 (generate_fixed_deals, evaluate_fixed)
```

`run_growing_pool.py`/`run_growing_pool_ppo.py` **never** call `train.py`/`train_ppo.py`'s `main()` functions directly: they launch each segment as a real subprocess (`subprocess.run([sys.executable, 'train.py', ...])`), to guarantee no state leaks from one segment to the next.

---

## 2. Conventions to know before reading the code

- **Cards**: a card is represented as text by `f"{rank}{suit}"`, e.g. `"AP"` = Ace of Spades, `"10K"` = 10 of Diamonds. Suits (`SUITS`, `coinche/game.py:4`) use the French suit initials: `P` = Pique (Spades), `C` = Cœur (Hearts), `K` = Karreau/Carreau (Diamonds), `T` = Trèfle (Clubs). Ranks (`RANKS`): `A, 10, K, Q, J, 9, 8, 7` (note: this `K` is the King rank, distinct from the Diamonds suit letter above).
- **Rank order**: `TRUMP_ORDER = [J,9,A,10,K,Q,8,7]` (trump) vs `NORMAL_ORDER = [A,10,K,Q,J,9,8,7]` (non-trump suit). A contract's `trump` is either a suit (`P/C/K/T`), `'SA'` (sans atout, no-trump — everything in `NORMAL_ORDER`), or `'TA'` (tout atout, all-trump — everything in `TRUMP_ORDER`).
- **Seats**: 0=North, 1=West, 2=South, 3=East (`SEAT_LABELS` in `coinche/players.py`, aligned with `PLAYER_POSITIONS` in `render_history.py`). Seats 0/2 are one team, 1/3 the other team (partner = `(seat + 2) % 4`). In **all** the RL code (`train.py`, `train_ppo.py`, `eval_*.py`), the trained/evaluated policy **always** occupies seats 0 and 2 (`RL_SEATS = (0, 2)`), the opponent seats 1 and 3.
- **Bid**: a tuple `(level:int, trump:str, coinched:bool, capot:bool)`, or `None` to pass. `level` is a multiple of 10 between 80 and 250 (250 = capot).
- **Game history (`GameEngine.history`)**: a dict `{'deal_hands', 'auction', 'tricks', 'contract', 'raw_points'}` — the only channel through which `HeuristicPlayer`/`HumanPlayer`/the neural network observe what's already happened in the donne (nobody "cheats" by reading `GameEngine.hands` of other seats directly, except the RL critic, which does it *deliberately* during simulation, see §4.4).
- **`torch is None`**: every RL module tests this condition and raises `SystemExit` with a clear message rather than crashing with an obscure `ImportError`, if `torch` isn't installed.

---

## 3. Step-by-step walkthrough (commands)

This section gives, in the order they're actually used, the concrete commands. All run from the repo root, with the `env_coinche` conda environment active.

### 3.1 Check that the game engine and the heuristic bidder run

```bash
python -m coinche.players
```
Runs `coinche/players.py` as a script (`if __name__ == '__main__': main()`, `coinche/players.py:1194`): deals a fixed hand to a `HeuristicPlayer` and prints the bid it produces. Used as a smoke test after a change to the bidding logic — no assertion, just a visual inspection of the result.

### 3.2 Play an isolated donne (no human interaction) and keep a JSON trace

```bash
python play_test.py --strategies random,random,random,random --out game_test.json
python play_test.py --strategies heuristic,heuristic,heuristic,heuristic --out game_test.json
python play_test.py --strategies heuristic,random,heuristic,random --hands-file my_hands.json --dealer 0 --out game_test.json
```
`--strategies` takes 4 values (seats 0 to 3) among `random`, `heuristic` (or `heuristic:variant`), `rl`/`agent` (⚠️ **untrained random** policy — `SimplePolicy`, not a real checkpoint) or `user`. `--hands-file` forces a specific donne via a JSON file `[["AP","10P",...], [...], [...], [...]]`. This is the simplest entry point to check the engine doesn't crash after a change to `game.py`/`players.py`.

### 3.3 Visualize a saved game as HTML

```bash
python3 render_history.py --history game_test.json --out game_history_viewer.html
```
Turns the JSON produced above into a standalone HTML page (tabs: auction + one per trick), to open in a browser. Useful to visually debug a contentious hand rather than reading raw JSON.

### 3.4 Play a donne on the keyboard, with AI support

```bash
python play_interactive.py --strategies heuristic,heuristic,human,heuristic
python play_interactive.py --strategies heuristic,heuristic,human,heuristic --advisor heuristic
python play_interactive.py --strategies rl:big:rl_experiments_v4/exp_growing_pool_bignet/seg_750000.pt,heuristic,human,heuristic
```
A `human` seat (South/seat 2 by default) reads its moves from the keyboard; an `--advisor` (an independent bot, never in `engine.players`) shows what it would have played on every human decision, without affecting the real game. `rl:architecture:path` loads a **real** trained checkpoint (unlike `play_test.py --strategies rl`). Generates a JSON history + an HTML replay at the end, and offers to replay the same donne with bots instead of the human for comparison.

### 3.5 Imitation pretraining (the starting point for any RL fine-tuning)

```bash
python pretrain.py --games 3000 --epochs 20 --out imit.pt
python pretrain.py --games 3000 --epochs 20 --architecture big --out imit_big.pt
```
Has 4 `HeuristicPlayer`s play against each other (attack and defense combined), collects `(state, chosen card)` at every decision, then trains a `CardNet` (or `CardNetBig` with `--architecture big`) by supervised classification to convergence. This is the `.pt` file later loaded with `--load` in `train.py`/`train_ppo.py`.

**A pitfall to know about `--architecture big`** (`CLAUDE.md`): `pretrain.py` has no early-stopping flag, so the command above trains `CardNetBig` **to full convergence** — and an `imit_big.pt` obtained this way measurably **hurts** downstream RL fine-tuning (a policy too peaked/low-entropy to explore). The `CardNetBig` checkpoint actually used in the validated runs (`rl_experiments_v4/imit_bignet_ent02.pt`) was produced by an ad hoc script, outside `pretrain.py`, stopped at a target entropy (~0.20) — not by this command as-is. The `imit_big.pt` generated here is fine for a smoke test of the `big` architecture, not for a real growing-pool run (see §3.9).

### 3.6 PPO critic pretraining (optional but recommended before PPO)

```bash
python pretrain_value.py --policy imit.pt --games 20000 --epochs 60 --out value_imit.pt
```
Has `imit.pt` play at seats 0/2 against `HeuristicPlayer` at seats 1/3, computes each donne's **counterfactual** return (the true target PPO trains on), and regresses a `ValueNet` on it via MSE. Load it afterward with `train_ppo.py --load-value`.

### 3.7 REINFORCE fine-tuning

```bash
python train.py --load imit.pt --episodes 100000 --lr 3e-5 --opponent heuristic --save exp1/final.pt
python train.py --load imit.pt --opponent heuristic,exp1/final.pt --pfsp --episodes 100000  # pool + PFSP
```
Trains the policy (shared between seats 0/2) against `--opponent` (a fixed opponent, a frozen copy of an old checkpoint, or a pool with uniform/PFSP sampling). The reward signal used for the update is a counterfactual gap, not the raw score (see §4.5).

### 3.8 PPO fine-tuning (an alternative to REINFORCE)

```bash
python train_ppo.py --load imit.pt --load-value value_imit.pt --episodes 100000 --lr 3e-5 --opponent heuristic
```
Same task/state/counterfactual as `train.py`, a different learning mechanism (batch + critic + clipping, see §4.6). `--architecture` picks between independent networks (`small`/`big`) or a shared actor/critic trunk (`shared`/`shared_aux`/`shared_deep`).

### 3.9 Growing-pool self-play — REINFORCE (the only structural idea that reliably wins)

```bash
python run_growing_pool.py --total-episodes 500000 --segment-episodes 50000 --init-load rl_experiments_v4/imit_bignet_ent02.pt --out-dir rl_experiments_v4/exp_growing_pool_bignet --architecture big
```
Splits training into successive `train.py` segments; each produced checkpoint joins the next segment's opponent pool (with PFSP). It's the orchestrator, not direct training — see §4.9 for the pool-pruning options (`--pool-keep-every`, `--pool-max-size`, `--pool-exclude`).

**`--init-load` must be a checkpoint of the same architecture as `--architecture`**: with `--architecture big`, `--init-load` must point to a `CardNetBig` state (e.g. `rl_experiments_v4/imit_bignet_ent02.pt` above, see the §3.5 pitfall — definitely not the `imit_big.pt` generated as-is by `pretrain.py --architecture big`, and even less `imit.pt`, which is a `CardNet` and would make loading fail with a missing/unexpected-keys error); with `--architecture small` (the default), `--init-load` must be a regular `imit.pt`.

### 3.10 Growing-pool self-play — PPO

```bash
python run_growing_pool_ppo.py --total-episodes 300000 --segment-episodes 50000 --seed-start 20
```
PPO equivalent of 3.9, drives `train_ppo.py` instead of `train.py`.

### 3.11 Evaluate one or more checkpoints

```bash
python eval_policy.py --games 45000 heuristic imit.pt exp_foo/final.pt
python eval_matchup.py --games 10000 heuristic imit.pt big:exp_foo/final.pt shared:exp_bar/final.pt
```
`eval_policy.py`: each checkpoint (greedy mode) against `HeuristicPlayer` alone, on exactly the same set of donnes fixed for everyone — a fair comparison. `eval_matchup.py`: round-robin, each checkpoint against **every other** (needed to judge the versatility of a checkpoint trained via self-play, which `eval_policy.py` alone can underestimate — see §4.12).

### 3.12 A typical end-to-end pipeline

A simplified diagram (not copy-pasteable commands as-is — the full flags are given in subsections 3.5/3.9/3.11/3.4 above). Step 1 is **not** a plain `pretrain.py` command for `--architecture big` — see the pitfall documented in §3.5 (a fully-converged `imit_big.pt` hurts downstream RL; the real starting point, `rl_experiments_v4/imit_bignet_ent02.pt`, comes from an ad hoc early stop on entropy, outside `pretrain.py`):

```
[ad hoc script, stopped at a target entropy ~0.20] → rl_experiments_v4/imit_bignet_ent02.pt
        │
        ▼
run_growing_pool.py --init-load rl_experiments_v4/imit_bignet_ent02.pt --architecture big → segments seg_*.pt
        │
        ▼
eval_matchup.py heuristic rl_experiments_v4/imit_bignet_ent02.pt seg_500000.pt ...   (check real progress)
        │
        ▼
play_interactive.py --strategies rl:big:seg_500000.pt,heuristic,human,heuristic   (play against / with it)
```

---

## 4. Detailed reference, file by file

### 4.1 `coinche/game.py`

Pure rules engine, with no dependency on `players.py` or `torch` — the only file everything else transitively imports. Implements `regles_coinche.md` in full (card order, move legality, §5 scoring including coinche/capot/ten-rounding/TA scale).

**Module constants** (`SUITS`, `RANKS`, `TRUMP_ORDER`, `NORMAL_ORDER`, `TRUMP_POINTS`, `NORMAL_POINTS`, `SA_POINTS`, `game.py:4`): the raw tables from `regles_coinche.md` §3 (card order) and §5 (card value by contract type). Reused as-is by `players.py`, `rl_agent.py`, and `render_history.py` (direct import, no duplication).

#### `Card` (`game.py:14`)
A card = `(suit, rank)` + a `__repr__` that produces the `"AP"` text format. It's this text representation that flows through `GameEngine.history` (JSON-serializable); the `Card` objects themselves only live for the current donne (in the players' hands).

#### `Deck` (`game.py:22`)
- `__init__`: builds the 32 cards (Cartesian product `SUITS × RANKS`).
- `shuffle()`: shuffles in place via `random.shuffle`.
- `deal() -> List[List[Card]]`: shuffles then splits into 4 packets of 8. Called only by `GameEngine.deal()` when no hands are given explicitly.

#### `Contract` (`game.py:34`)
A passive `(level, trump, coinched, capot)` structure — no logic, just a container. Built by `GameEngine.run_auction()` once the auction is over, read by `GameEngine.play()` for scoring and by everything else (`players.py`, `rl_agent.py`, `render_history.py`) via `engine.contract`.

#### `GameEngine` (`game.py:41`)
The heart of the engine — holds `players` (4 `Player` objects), `dealer`, `hands`, `contract`, `taker_idx`, `history`. One instance = one donne (not a full match to 3000 points).

- **`deal(hands=None)`** (`game.py:51`): deals the cards (either via `Deck().deal()`, or by parsing given strings, e.g. `["AP","10P",...]`), calls `player.deal(hand)` on each player (letting them keep a copy of their starting hand), assigns them `seat` and `engine` (it's this `setattr` that gives `HeuristicPlayer`/`RLPlayer`/`HumanPlayer` access to `self.engine.history`), and initializes `self.history`. Called by `run_auction()` on a redeal (everyone passes) and by every high-level script (`play_test.py`, `train.py`, `eval_policy.py`, ...).
- **`_normalize_bid(bid)`** (static, `game.py:84`): normalizes any capot bid to `(250, trump, coinched, True)`. Called by `run_auction()` on every raw bid returned by `player.bid()`, and by `HumanPlayer.bid()` before validating keyboard input.
- **`_is_valid_bid(bid, current_best)`** (static, `game.py:92`): checks the level is a multiple of 10 within `[80,250]`, the coinche rule (must exactly reproduce the last bid, not outbid it), the normal-raise rule (`+10` minimum). Called by `run_auction()` to validate/reject every bid, and by `HumanPlayer.bid()` to re-prompt on invalid input.
- **`run_auction()`** (`game.py:110`): loops over seats starting at `(dealer+1)%4`, calls `player.bid(current_best)` on each, until 4 consecutive passes or a coinche (which immediately closes the auction). If everyone passes, increments `dealer` and **recursively calls `self.deal()` then `self.run_auction()`** — it's this behavior that traps naive evaluation scripts (cf. `eval_policy.py::generate_fixed_deals`, §4.11, which filters this case out). Sets `self.contract`/`self.taker_idx` and feeds `history['auction']`/`history['contract']`.
- **`card_order_key(card, lead_suit, trump)`** (`game.py:158`): returns a tuple `(is_trump, is_lead, -rank_idx)` comparable with plain Python operators — a card's "strength" in the context of a given trick. Used by `evaluate_trick`, `legal_moves`, and conceptually copied (same logic, reimplemented) in `HeuristicPlayer._rank_strength` and `rl_agent._current_trick_winner_seat`.
- **`evaluate_trick(trick, trump)`** (`game.py:171`): applies `card_order_key` to every card in the trick and returns the winning seat. Called once per trick in `play()`.
- **`legal_moves(seat, hand, trick, trump)`** (`game.py:179`): implements `regles_coinche.md` §4 (follow/cut/overtrump/discard), with specific handling for `SA` (follow if possible, otherwise a free discard) and for `TA` (an obligation to overtrump in the suit led, treated as a trump led — this is the bug fixed on 2026-07-29, see `CLAUDE.md`). **Never validates itself** what a player returns: it's up to each `Player.play_card()` to call it and choose among its result. Called by `HeuristicPlayer.play_card()`/`RLPlayer.play_card()`/`HumanPlayer.play_card()`/`NeuralPolicy.choose_card()`/`PPOPolicy.choose_card()` — never by `GameEngine.play()` itself.
- **`card_point(card, trump)`** (`game.py:243`): a card's point value according to `TRUMP_POINTS`/`SA_POINTS`/`NORMAL_POINTS`. Called by `play()` for the final score, and deliberately duplicated (same table, no direct import of instance methods) in `render_history.card_point`, while `rl_agent._points_so_far` (via `engine.card_point`) *does* reuse it.
- **`_round_dizaine(points, direction)`** (static, `game.py:252`): rounds to the nearest ten — `'nearest'` (standard, suit/SA contracts), `'down'`/`'up'` (the attack/defense asymmetric rounding specific to TA). Called only by `play()`.
- **`play()`** (`game.py:265`): plays the 8 tricks (loops `player.play_card()` by seat in order, starting from the previous trick's leader), then computes the score according to `regles_coinche.md` §5 in full — capot, falling short, ten-rounding, TA scale (÷15 instead of direct rounding), belote (credited to the team that wins the trick containing the 2nd of the two trump King/Queen cards, not necessarily to whoever held them), then doubles everything via coinche if `contract.coinched`. Returns `(team_points: dict[0|1, int], contract)`. This is the function `CoincheEnv.step()`, `play_test.run_game()`, `train.run_episode()`/`counterfactual_reward()`/`evaluate()`, and every eval script call to get a donne's final result.

### 4.2 `coinche/players.py`

All the player strategies, plus a few text-formatting helpers used by `HumanPlayer` and `play_interactive.py`. Everything inherits from `Player` (the common `deal`/`bid`/`play_card` interface).

**Module helpers** — `count_suit(hand, suit)` and `has_rank(hand, suit, rank)` (`players.py:8-12`): two trivial predicates reused throughout almost all of `HeuristicPlayer`'s bidding and play logic (counting/testing the hand). `RANK_ORDER` (`players.py:6`) is defined but unused elsewhere in this file (a leftover, no impact).

#### `Player` (`players.py:14`)
The base interface: `deal(hand)` copies the hand into `self.hand`/`self.initial_hand` and resets `self._raise_contributions` (used only by `HeuristicPlayer`, see below); `bid()` returns `None` by default (pass); `play_card()` raises `NotImplementedError`. `GameEngine.deal()`/`play()` only ever call these three methods — any object that implements them can play a game.

#### `RandomPlayer` (`players.py:31`)
A naive baseline: `bid()` passes 70% of the time, otherwise announces a random level/suit with no strategic coherence at all; `play_card()` follows the suit led if possible, otherwise cuts if possible, otherwise plays anything — no use of `legal_moves()`, so it **can play an illegal move** with respect to the overtrump obligation (this is intentional: it serves as the weakest possible baseline for smoke tests, not a credible AI).

#### `HeuristicPlayer` (`players.py:59`) — implements `heuristiques.md`

This is the densest class in the repo. It's organized into two blocks, aligned with `heuristiques.md`'s numbering.

**Bidding block (heuristiques.md §1.x)**

- `_last_bid_info()` (`players.py:68`): the last offer *actually made* (not passes) in the current auction, read from `engine.history['auction']`. Called first thing by `bid()` to know whether the context is "I'm responding to my partner" or "to an opponent".
- `_count_tricks(trump_suit, hi, lo, third)` (`players.py:78`): estimates the number of winnable tricks in the current hand — parametrized to count in a suit contract (`hi='A', lo='10'`), in SA, or in TA (`hi='J', lo='9', third='A'`, called with `trump_suit=None`). Called by `bid()` (suit candidates), `_partner_sa_remonte`/`_partner_ta_remonte` (checking that at least 4 real tricks back a raise), and `_can_coincher`.
- `_decrochage_ok(own_level, opp_level)` (`players.py:101`): tolerates a "small lie" of one level during a décrochage. Called only within `bid()`.
- `_my_last_bid()` / `_last_partner_bid()` (`players.py:107`, `118`): my last offer / my partner's last offer in the current auction, read from the history. `_last_partner_bid` is used in `bid()` when an opponent spoke after the partner (I must still be able to raise the partner's offer, not just décrocher on my own).
- `_team_last_bidder_is_me()` (`players.py:132`): true if, between my partner and me, I'm the last one to have actually bid. Forbids décrochage (`bid()`) when my partner hasn't taught me anything since my last bid.
- `_decrochage_adjusted_level(bidder_seat, bid)` (`players.py:152`): if the bid being examined was itself a décrochage on a lower-level opposing offer, returns the real implicit level (`level - 10`). Called by `_sa_known_aces` and `_combined_count` to avoid over-interpreting a level inflated by décrochage.
- `_sa_known_aces(bidder_seat, bid)` (`players.py:171`): a `{80:2, 90:3, 100:4}` table applied to the adjusted level. Called by `_partner_sa_to_color` and `_combined_count`.
- `_my_earlier_bid_of_type(bid_type)` (`players.py:175`): my own first SA/TA offer of this auction, if it precedes the partner's current raise — avoids double counting. Called by `_combined_count`.
- `_partner_sa_to_color(partner_seat, partner_bid, own_color_candidates)` (`players.py:190`): switches to my own suit rather than simply supporting the partner's SA, credited with the aces they announced (minus one). Called by `bid()` right after `_partner_sa_remonte`.
- `_partner_color_remonte(partner_bid)` (`players.py:205`): raises a suit bid from the partner (trump support at the first level, then outside aces). Called by `bid()` when `partner_bid[1]` is a suit.
- `_combined_count(partner_bid, bid_type, my_count)` (`players.py:244`): adds up the master cards (aces/jacks) already announced by the partner (via the adjusted level) and my own, without double counting if their raise comes from my own earlier bid. Called by `_partner_sa_remonte`/`_partner_ta_remonte`.
- `_partner_sa_remonte(partner_bid)` / `_partner_ta_remonte(partner_bid)` (`players.py:258`, `280`): raises a partner's SA/TA — my aces/jacks, then a bonus for non-third tens/nines if the team has ≥4 master cards **and** ≥4 real tricks counted (a guard against overvaluing). Called by `bid()`.
- **`bid(current_best)`** (`players.py:300`): the entry point — builds a list of `candidates` (suit §1.1, SA §1.2, TA §1.3, partner's raise), picks the max (at an equal level, prefers extending the partner's bid over a solo bid via `_bid_key`), applies décrochage against an opponent (never the partner), then evaluates coinche (`_can_coincher`) as a last resort if no outbid is justified. This is the method `GameEngine.run_auction()` calls on every turn; `RLPlayer` inherits it as-is (no substitution — only `play_card` changes).
- `_needed_tricks_to_coincher(level)` (`players.py:436`): a decreasing scale of the number of tricks required on defense to coinche (§1.4). Called by `_can_coincher`.
- `_defense_trump_tricks(trump)` (`players.py:443`): estimates how many tricks my own trumps are worth *on defense* (more pessimistic than on attack: any 3 trumps count for nothing on their own). Called by `_can_coincher`.
- `_can_coincher(opponent_bid)` (`players.py:458`): combines `_defense_trump_tricks` + off-trump `_count_tricks` (suit), or `_count_tricks(None, ...)` (SA/TA), and compares against `_needed_tricks_to_coincher`'s scale. Called last within `bid()`.

**Card-play block (heuristiques.md §2.x)**

- `_rank_strength(card, trump, lead_suit)` (`players.py:476`): the equivalent of `GameEngine.card_order_key` but reimplemented on the player side (no `self.engine` access needed) — used everywhere in this block to compare two cards.
- `_master_ranks(trump)` (`players.py:488`): `('J','9')` in TA, `('A','10')` otherwise — the (master card, second card) pair according to the contract type. Called at the top of `play_card()` then passed to nearly every method in this block.
- `_weakest`/`_strongest(cards, trump, lead=None)` (`players.py:492`, `495`): min/max in the sense of `_rank_strength` — utilities called in nearly every method below.
- `_long_suit_lead(cards, trump, master_lo)` (`players.py:498`): in a long suit without the ace/jack but holding the 10/9, picks the worst card of the contiguous run immediately following the 10/9 (keeping the 10/9 for later). Called by `_lead_attack_sa_ta`.
- `_played_cards()` / `_card_seen_before(suit, rank)` (`players.py:515`, `522`): the list/test of cards already played in complete tricks, read from `engine.history['tricks']`. Used almost everywhere in this block and by `rl_agent._played_before_this_trick` (equivalent logic, reimplemented on the state-encoding side).
- `_trumps_remain_with_opponents(trump)` / `_trump_already_led(trump)` (`players.py:525`, `530`): is there any trump left anywhere but my hand? has trump already been led at least once? Used by `_lead_taker_trump`/`_lead_partner_trump` to decide whether to "draw" trump or hold it back.
- `_attaquant_principal_seat()` (`players.py:535`): finds who **initiated** the retained contract type (not necessarily `engine.taker_idx`, which is only the author of the very last bid) — a necessary distinction since `heuristiques.md` treats the "main taker" and their following partner differently. Called by `play_card()` (to set `is_taker`) and by `_is_coincheur_before_taker`.
- `_color_contract_coinched()` / `_coincheur_seat()` / `_is_coincheur_before_taker()` (`players.py:556`, `562`, `575`): §2.5's logic (a coinched hand) — identifies whether the defender who called the coinche themselves sits right before the main taker, the case where the defense must prioritize leading its long suit above all else. Called by `_lead_defense`.
- `_is_confirmed_master(card, trump)` (`players.py:589`): true if every card that outranks `card` in its suit has already been played (so it's a master even without literally being the ace/10). Called by `_lead_offsuit_master`, `_lead_defense`, `_lead_attack_sa_ta`.
- `_choose_attack_suit(trump)` (`players.py:597`): the suit where the attack initially held the most cards (based on `self.initial_hand`, not the current hand). Called by `_lead_offsuit_master` and `_lead_coinche_defense_long_suit`.
- `_suit_to_replay_for_partner(master_hi)` (`players.py:608`): if my partner opened a suit for the first time and I won it with my master card, finds that suit to lead it again as a priority. Called by `_lead_attack_sa_ta`.
- `_lead_offsuit_master(trump, master_hi, master_lo)` (`players.py:632`): off-trump lead logic on the attack side (suit contract) — master card first, then the second card if the ace has been played, then any card confirmed as master by elimination, otherwise the weakest card of the chosen attack suit. Called by `_lead_taker_trump`/`_lead_partner_trump`.
- `_lead_coinche_defense_long_suit(trump, master_hi, master_lo)` (`players.py:649`): §2.5, absolute priority to the long suit (ace first if held) to force the taker to cut. Called by `_lead_defense`.
- `_lead_taker_trump(trump, master_hi, master_lo)` / `_lead_partner_trump(trump, master_hi, master_lo)` (`players.py:663`, `684`): trump lead on the main taker's side vs the following partner's side — the jack-first logic, the "second 9" rule (holding it back on the first trump round), and stopping "drawing" trump once we know we're the only one still holding any. Called by `_lead_card`.
- `_lead_defense(trump, master_hi, master_lo)` (`players.py:700`): lead on the defense side — the priority coinche case (`_is_coincheur_before_taker`), otherwise an off-trump master card, a second card that's become master, a card confirmed master by elimination, a singleton, otherwise the weakest. Called by `_lead_card` and `_lead_attack_sa_ta` (the SA/TA defense case).
- `_prefer_9_second_suit(candidates, trump, is_taker)` (`players.py:728`): in TA, the taker's partner (not the taker) prefers to lead in a suit where they have a second 9, a signal for their partner. Called by `_lead_attack_sa_ta`.
- `_lead_attack_sa_ta(trump, master_hi, master_lo, is_taker)` (`players.py:740`): the longest lead method — priority to suits that have become master by elimination, then replaying the partner's opening suit if I'm master there, then a ranking of long-with-master / long-without-master / short suits. Called by `_lead_card` (SA/TA case, attack side).
- `_lead_card(trump, is_attacker, is_taker, master_hi, master_lo)` (`players.py:804`): routes between the 5 lead methods above based on `trump`/`is_attacker`/`is_taker`. Called only by `play_card()` when the trick is empty (`not trick`).
- `_is_absolute_master_rank(trump)` / `_partner_is_absolute_master(current_winner, trump)` (`players.py:815`, `818`): the absolutely unstoppable rank (jack for trump/TA, ace for SA); tests whether the trick's current master (my partner) holds this unstoppable card. Called by `_discard` and `_follow_card`.
- `_choose_defausse_suit()` (`players.py:831`): weak suit = an initial singleton, or an initial doubleton with no 10 or ace (`heuristiques.md` §2.3) — based on `self.initial_hand`, a random pick if there are several tied candidates (`HeuristicPlayer`'s only source of randomness outside bidding — hence the pitfall documented in `eval_policy.py`, see §4.11). Called by `_discard`.
- `_discard(trump, current_winner)` (`players.py:844`): discards in the chosen weak suit; if the partner is already the trick's unstoppable master, gets rid of the strongest non-master card rather than the weakest (secures without wasting a useful card). Called by `_follow_card` when I can neither follow nor cut.
- `_secure_when_partner_wins(same, trump, lead, trick)` (`players.py:854`): when the trick is already won for my side, decides which of my cards in the suit led to play now risk-free (keeping the best one if it's not secured yet, securing the 2nd best otherwise). Called by `_follow_card`.
- `_follow_card(trick, trump, is_attacker, master_hi, master_lo)` (`players.py:874`): the follow method — the "I can follow the suit led" case (with a special "drawing trump" sub-case on the attack side of a suit contract), "I must cut/overtrump", and otherwise delegates to `_discard`. Called only by `play_card()` when the trick isn't empty.
- **`play_card(seat, leader, trick, trump)`** (`players.py:939`): computes `is_attacker`/`is_taker` then delegates to `_lead_card` or `_follow_card` depending on whether the trick is empty, removes the chosen card from `self.hand` and returns it. This is the method `GameEngine.play()` calls for every card; `RLPlayer.play_card()` falls back to it when no policy is attached.

#### Text-formatting helpers (`players.py:957-1021`)
`SEAT_LABELS`, `SUIT_SYMBOLS`, `_trump_label`, `_fmt_card`, `_fmt_hand`, `_fmt_bid`, `_parse_bid_str`, `_find_card`, `_card_from_repr` — trivial text ↔ `Card`/bid conversions for display and keyboard input. They exist only for `HumanPlayer` (and `play_interactive.py`, which imports `SEAT_LABELS`); no game logic depends on them.

#### `HumanPlayer` (`players.py:1025`)
A keyboard-driven player. `deal(hand)` prints "new deal" after the first one (a redeal case). `_advisor_hint(kind, ...)` (`players.py:1045`) queries an independent `Player` (`self.advisor`, never inserted into `engine.players`) about what it would have chosen, resyncing a **copy** of the hand on every call so the real hand is never mutated. `show_new_tricks()`/`_print_trick_recap()` (`players.py:1078`, `1070`) catch up the display of finished tricks since the last call (needed since this player may not be the last to play in a trick). `bid()`/`play_card()` (`players.py:1096`, `1140`) display the current context (bids so far / trick state, sorted hand) then read stdin in a loop until valid input, reusing `engine._is_valid_bid`/`engine.legal_moves` for legality. Called only by `GameEngine.run_auction()`/`play()` like any other `Player`.

#### `RLPlayer(HeuristicPlayer)` (`players.py:1179`)
Inherits `bid()` from `HeuristicPlayer` unmodified (bidding always stays heuristic). Only overrides `play_card()`: delegates to `self.policy.choose_card(...)` if a policy is attached (an interface common to `SimplePolicy`, `NeuralPolicy`, `PPOPolicy`, `SharedTrunkPPOPolicy`, `SharedTrunkAuxPPOPolicy` — all defined in `rl_agent.py`/`train_ppo.py`), otherwise falls back to the inherited heuristic play. This is the seat used at positions 0/2 in `train.py`/`train_ppo.py`/`eval_policy.py`/`eval_matchup.py`.

#### `create_player(strategy, name)` (`players.py:1196`)
A factory from a string (`'random'`, `'human'`, `'heuristic'`/`'heuristic:variant'`/`'user'`, `'rl'`/`'agent'`, otherwise `RandomPlayer` by default). Used by `play_test.py::build_players` and `play_interactive.py::build_one_player` — **not** by `train.py`/`eval_policy.py`, which build their `RLPlayer` directly with a real loaded policy.

### 4.3 `coinche/env.py`

#### `CoincheEnv` (`env.py:5`)
A minimal Gym-like wrapper around a single donne (not a match to 3000 points). `__init__(players=None, dealer=0, agent_seat=0)` builds 4 `RandomPlayer`s by default if no players are given, and an internal `GameEngine`. `reset(hands=None)` (`env.py:15`) calls `engine.deal()` then `engine.run_auction()`, returns a minimal observation (`_get_obs`, `env.py:21` — just the current hand, not the trick). `step(action)` (`env.py:28`) actually ignores the `action` passed in (each player already decides for itself via `play_card`) and simply calls `engine.play()` to run all 8 tricks at once, returning `reward = team_points[agent_seat % 2]`. Used only by `play_test.py::run_game` — neither `train.py` nor `train_ppo.py` go through this class (they call `GameEngine` directly in `run_episode`), because its "one action = one RL decision" model doesn't fit their need to replay a counterfactual with all 4 seats controlled by the same opponent.

### 4.4 `coinche/rl_agent.py`

Everything that turns a game state into a numeric vector, and all the neural networks. Degrades to `torch = None` if `torch` isn't installed (every `nn.Module` class is then undefined, guarded by `if torch is not None:`).

**Low-level encoding functions**

- `_canonical_slots(trump)` (`rl_agent.py:14`): computes 4 canonical suit "slots", invariant under rotation of the real trump suit — slot 0 is always trump (suit contract) or any suit (TA), with the associated rank order (`TRUMP_ORDER`/`NORMAL_ORDER`). This is the pivotal function: everything else in the file (and `train_ppo.py`) calls it to never reason in terms of absolute physical suit — the network only needs to learn the "best trump card" concept once, not 4 times (once per physical suit).
- `_slot_index(suit, rank, suits, orders)` (`rl_agent.py:36`): a given card's `0..31` position in the canonical space — used to build the multi-hot vectors and to decode the network's chosen action (index → real card) in `NeuralPolicy.choose_card`/`PPOPolicy.choose_card`/`pretrain.RecordingHeuristicPlayer`.
- `_relative_multi_hot(cards, suits, orders)` (`rl_agent.py:42`): a 32-dim vector, 1.0 at the positions occupied by `cards`. The basic building block reused to encode the hand, already-played cards, the current trick, and the initial hand.
- `_one_hot(index, size)` (`rl_agent.py:49`): a generic one-hot (suit led, position in the trick, trump type).
- `_played_before_this_trick(player, suits, orders)` (`rl_agent.py:56`): the equivalent (reimplemented, not called) of `HeuristicPlayer._played_cards`, but directly returns a canonical multi-hot vector rather than a list of strings.
- `_slot_counts(vec32)` (`rl_agent.py:72`): the sum per 8-block of a 32-dim vector — used to derive `length_vec` (initial suit length) and `unknown_vec` (still-unknown cards) in `encode_state`.
- `_record_void_from_trick(seat_suit_pairs, void, suits)` / `_void_vec(player, trick, suits)` (`rl_agent.py:78`, `93`): detects certain voids (a seat that didn't follow the suit led when it should have) and produces a 12-dim vector (3 seats × 4 slots). `_void_vec` scans the complete tricks' history **and** the current trick.
- `_auction_signals_vec(player, suits)` (`rl_agent.py:119`): for each of the 3 other seats, did they bid/raise each suit (12-dim) or SA/TA (6-dim) at any point in the auction, even if the final contract ended up elsewhere — 18-dim total.
- `_led_suit_vec(player, trick, suits)` (`rl_agent.py:153`): for each of the 3 other seats, have they already led a trick in each suit (12-dim) — distinct from simply counting played cards, captures a specific seat's suit preference (the numeric analog of `HeuristicPlayer._suit_to_replay_for_partner`).
- `_points_so_far(player, trump, engine)` (`rl_agent.py:183`): card points already banked by my side / the opponent in complete tricks, normalized by 162. Calls `engine.card_point` (directly reuses `GameEngine`, no duplication of the point table here).
- `_current_trick_winner_seat(trick, trump, engine)` (`rl_agent.py:209`): the seat currently winning the trick in progress, reusing `engine.card_order_key` — used to derive `partner_winning` in `encode_state`.

**`encode_state(player, trick, trump, ablate_points=False)`** (`rl_agent.py:223`, `STATE_DIM = 167`): assembles every block above into a single fixed vector — it's **exactly** what a real player can know (never the other hands). Called by `NeuralPolicy.choose_card`, `PPOPolicy.choose_card`, `SharedTrunkPPOPolicy.choose_card`, `SharedTrunkAuxPPOPolicy.choose_card`, `pretrain.RecordingHeuristicPlayer.play_card` (collecting the imitation dataset), and internally by `encode_full_state`. `ablate_points` is an ablation-study flag (forces `points_vec` to zero without changing `STATE_DIM`) propagated from `pretrain.py --ablate-points-so-far` / `train_ppo.py --ablate-points-so-far`.

- `_other_hands_vec(player, suits, orders)` (`rl_agent.py:312`): the **current** hands of the 3 other seats — privileged information, read directly from `engine.players[other_seat].hand` (deliberately "cheating" access, possible only because we simulate the whole donne during training). Called only by `encode_full_state`.
- `_other_trump_counts(player, suits)` / `other_trump_counts(player, trump)` (`rl_agent.py:328`, `347`): the number of trumps currently in hand for the 3 other seats — `SharedTrunkActorCriticAux`'s auxiliary head target. `other_trump_counts` is the public version (computes `suits` itself), called by `SharedTrunkAuxPPOPolicy.choose_card` in `train_ppo.py`.
- **`encode_full_state(player, trick, trump, ablate_points=False)`** (`rl_agent.py:354`, `FULL_STATE_DIM = 263`): `encode_state(...)` + `_other_hands_vec(...)` — the **centralized** state only the critic sees (never the actor). Called by `PPOPolicy.choose_card`/`SharedTrunkPPOPolicy.choose_card`/`SharedTrunkAuxPPOPolicy.choose_card` (in `train_ppo.py`) and by `pretrain_value.collect_dataset`.

**`SimplePolicy`** (`rl_agent.py:371`): a random policy among legal moves, without `torch`. It's the fallback used by `play_test.py`/`play_interactive.py` for the `'rl'`/`'agent'` strategy **when no checkpoint is loaded** — also serves as a progress reference ("does the trained policy at least beat random play?").

**Networks (only defined if `torch is not None`)**

- `CardNet` (`rl_agent.py:383`): 1 hidden layer (167→128, ReLU) + 1 output (128→32 logits, one per canonical slot). The reference architecture, historically validated — the one `pretrain.py` produces by default.
- `CardNetBig` (`rl_agent.py:393`): 4 layers (167→128→128→64→32), Pre-LN LayerNorm before each linear layer + GELU. Tests whether a single hidden layer is a capacity ceiling for `CardNet`. A single `forward` (no separate head) — the `__init__`/`forward` methods follow the same mechanical pattern as `CardNet`, just with 4 layers instead of 2.
- `SharedTrunkActorCritic` (`rl_agent.py:421`): a shared actor/critic trunk (167→128→128, same parameter names as `CardNetBig`'s first 2 layers, deliberately so an existing `imit_bignet*.pt` can be reused). Methods: `_trunk(x)` (the 2 shared layers), `forward_actor(x)` (trunk + private actor head 128→64→32), `forward(x)` (an alias for `forward_actor`, so `train.py`'s `NeuralPolicy`/`make_opponent_factory` can load this network as a frozen opponent with no code change), `forward_critic(x_full)` (splits `x_full` into the visible state + 96-dim centralized info, runs the visible state through the shared trunk, the centralized info through a small dedicated `fc_central` branch, concatenates, then a private critic head), `load_actor_from_cardnetbig(path)` (remaps an existing `CardNetBig` checkpoint's keys onto this network — only the critic part starts out random).
- `SharedTrunkActorCriticAux(SharedTrunkActorCritic)` (`rl_agent.py:524`): adds `forward_aux(x)`, a 3rd head that predicts the number of trumps for the 3 other seats **from the shared trunk alone** (never from the critic's centralized branch — a deliberate choice to prevent the network from "cheating" by reading the answer out of that branch instead of encoding it into the trunk, which would empty the auxiliary task of its purpose).
- `SharedTrunkActorCriticDeep` (`rl_agent.py:559`): a rebalancing — a 3-layer trunk (167→128→128→64, covering exactly `CardNetBig`'s first 3 layers), a single-layer actor head (64→32, `CardNetBig`'s last one). An independent class (not a subclass of `SharedTrunkActorCritic`, the trunk shape differs too much). Same `_trunk`/`forward_actor`/`forward`/`forward_critic`/`load_actor_from_cardnetbig` methods as `SharedTrunkActorCritic`, adapted to a 3-layer trunk.

**`NeuralPolicy`** (`rl_agent.py:655`) — the REINFORCE policy, used by `train.py` and as a loadable frozen opponent (`make_opponent_factory`):
- `__init__(device, lr, baseline_beta, entropy_beta, net_cls=CardNet)`: builds `net_cls()` + an Adam optimizer, initializes the trajectory buffers.
- `choose_card(player, legal, leader, trick, trump)` (`rl_agent.py:674`): encodes the state (`encode_state`), masks illegal moves' logits to `-inf`, samples (`record=True`, training mode — stores log-prob and entropy) or takes the argmax (`record=False`, greedy evaluation mode), then decodes the chosen index into a real `Card` via `suits`/`orders`. Called by `RLPlayer.play_card`.
- `update(reward)` (`rl_agent.py:695`): one REINFORCE update per donne — `loss = -(sum of the trajectory's log-probs) * advantage - entropy_beta * entropy`, where `advantage = reward - baseline` (baseline = a global moving average). Called by `train.py::main` after every episode (with the *counterfactual* reward, not the raw one — see §4.5).
- `save(path)` / `load(path)` (`rl_agent.py:714`, `717`): `save` just writes the network's `state_dict`; `load` accepts either this format or a native PPO checkpoint (dict with a `'policy'` key) — needed so `make_opponent_factory` (train.py) can load a checkpoint produced by `train_ppo.py` as a frozen pool opponent.

### 4.5 `train.py`

REINFORCE training. Imports `GameEngine` (game.py), `RLPlayer`/`HeuristicPlayer` (players.py), `NeuralPolicy`/`CardNet`/`CardNetBig` (rl_agent.py). `RL_SEATS = (0, 2)` (`train.py:34`, documentation, not hard-used elsewhere except by convention in the following code).

- `_default_opponent(name)` (`train.py:37`): builds a `HeuristicPlayer` — the whole file's default factory.
- **`make_opponent_factory(token, net_cls=CardNet)`** (`train.py:41`): converts an `--opponent` token (`'heuristic'` or a checkpoint path) into a `name -> Player` function. For a path, loads a frozen `NeuralPolicy(net_cls=net_cls)` (`record=False`) and returns a `lambda` that builds an `RLPlayer` around this shared policy (the same object reused for both of the factory's calls — so a single network loaded in memory even though the opponent occupies 2 seats). Called by `build_opponent_pool` and reused as-is by `train_ppo.py` (direct import, no duplication).
- **`build_opponent_pool(opponent_arg, net_cls=CardNet)`** (`train.py:59`): parses `--opponent` (`"token1,token2,..."`) into `[(token, factory), ...]`. Called by `main()` (train.py and train_ppo.py) and indirectly by `run_growing_pool.py`/`run_growing_pool_ppo.py` (they build the `--opponent` string passed to the subprocess, not this function directly).
- **`PFSPSampler`** (`train.py:94`): opponent sampling biased toward the toughest one (Prioritized Fictitious Self-Play, AlphaStar-style). `__init__(pool, refresh_every, temperature, ema_beta, explore_eps)` initializes a 0.5 EMA win rate for every opponent. `_refresh_weights()` (`train.py:107`) recomputes a softmax over `1 - win_rate` (adjustable temperature), mixed with a uniform exploration floor `explore_eps`. `choose(global_ep)` (`train.py:115`) refreshes the weights every `refresh_every` episodes then samples a weighted opponent. `record_outcome(name, reward)` (`train.py:126`) updates the win-rate EMA after every episode actually played against that opponent. `summary()` (`train.py:130`) formats a text summary (`"name:Nx(wr=0.xx)"`) — it's this line, printed at the end of training (`"PFSP picks total: ..."`), that `run_growing_pool.py::_parse_final_winrates` reparses to prune the pool by weakness.
- **`run_episode(policy, dealer, opponent_factory=_default_opponent)`** (`train.py:136`): plays a complete donne, policy at seats 0/2, `opponent_factory` at seats 1/3. Returns `(reward, deal_hands)` — `reward = team_points[0] - team_points[1]`. Called by `main()` (train.py and train_ppo.py) and by `evaluate()`.
- **`counterfactual_reward(hands, dealer, opponent_factory=_default_opponent)`** (`train.py:153`): replays **exactly the same hands** with `opponent_factory` at all 4 seats (no policy) — the reference signal for "what would this opponent have done in my place, on the same cards". Must always receive the same `opponent_factory` as the `run_episode` call that produced `hands` (otherwise two different opponents get compared). Called by `main()` (both training files) and by `pretrain_value.collect_dataset`.
- **`evaluate(policy, n_games, start_dealer=0, opponent_factory=_default_opponent)`** (`train.py:174`): switches `policy.record` to `False`, plays `n_games` greedy donnes via `run_episode`, restores `policy.record`, returns `(avg, win_rate)`. Called periodically by `main()` (in `train.py` and `train_ppo.py`) to log progress during training — always against `heuristic` by default, even during pool training, for comparable tracking across runs. This is *during-training* tracking, on donnes drawn at random on every call; the final "serious" evaluation of a checkpoint instead uses `eval_policy.py`/`eval_matchup.py`, which have their own `evaluate_fixed` with fixed donnes shared across checkpoints (see §4.11-4.12).
- `entropy_beta_for_episode(base_beta, ep, decay)` (`train.py:194`): computes the effective entropy coefficient according to the chosen schedule (`'none'`, `'invsqrt'`, `'inv'`). Called on every episode by `main()` (both files).
- **`main()`** (`train.py:207`): parses the CLI arguments, builds the opponent pool + optionally a `PFSPSampler`, loads `--load` if given, then loops over `--episodes`: picks an opponent (uniform pool or PFSP), plays the episode (`run_episode`), computes the training reward (raw or counterfactual depending on `--no-counterfactual-baseline`), calls `policy.update(training_reward)`, logs/evaluates/checkpoints periodically. This is the entry point run by `python train.py ...` **and** the one launched as a subprocess by `run_growing_pool.py`.

### 4.6 `train_ppo.py`

PPO training — the same task as `train.py` (same state, same seats, same counterfactual), a different mechanism: batching several donnes, a per-state critic, several clipped passes. Directly re-imports `build_opponent_pool`, `PFSPSampler`, `run_episode`, `counterfactual_reward`, `evaluate`, `entropy_beta_for_episode` from `train.py` (no duplication of this logic).

- **`ValueNet`** (`train_ppo.py:101`): a **centralized** critic (takes `encode_full_state`, `FULL_STATE_DIM`) with a trunk **independent** of the policy's (no sharing with `CardNet` — a shared trunk inherited from `imit.pt` turned out unable to separate attack/defense after 50k episodes, cf. `remarques_rl.md`). 1 hidden layer (128) + 1 scalar output. `forward(x)`: ReLU then the value head.
- **`PPOPolicy`** (`train_ppo.py:106`): implements the `choose_card`/`record`/`save`/`load` interface expected by `RLPlayer`/`run_episode`/`evaluate`, plus PPO-specific `end_episode`/`update_batch`.
  - `__init__`: `policy_net` (`CardNet`/`CardNetBig` per `policy_net_cls`) + `value_net` (`ValueNet`) under a **single** Adam optimizer (one `.step()` updates both networks, which nonetheless remain two separate trunks).
  - `choose_card(...)` (`train_ppo.py:138`): encodes the visible state (actor) and the full state (critic, `encode_full_state`), samples the action, and **stores** `(state, full_state, action, log_prob, value, mask)` in `self._traj` — unlike `NeuralPolicy`, nothing is updated immediately.
  - `end_episode(reward)` (`train_ppo.py:166`): archives `(trajectory, reward)` into `self._episodes` (no update). Called by `main()` in place of `.update()`.
  - `update_batch()` (`train_ppo.py:176`): concatenates every trajectory accumulated since the last call, computes the advantage (`return - predicted_value`, normalized) unless `no_critic_baseline` (then advantage = return centered/normalized on the batch, `value_net` neither called nor trained), then runs `self.epochs` minibatch passes with the clipped PPO objective (`clip_eps`) + MSE value loss (`value_coef`) + entropy bonus (`entropy_coef`). Returns the average `(policy_loss, value_loss, entropy, clip_frac)`. Called by `main()` every `--batch-episodes` episodes.
  - `save(path)`/`load(path)`/`load_value(path)` (`train_ppo.py:270`, `287`, `309`): `save` writes a `{'policy':, 'value':}` dict (**not** the optimizer's state — a resume restarts with a "cold" Adam). `load` accepts a native PPO checkpoint, a raw REINFORCE checkpoint (`CardNet`, same shape), or an older shared-trunk format (`policy_head`/`value_head`, for backward compatibility). `load_value` loads `value_net` separately (a native PPO checkpoint or `pretrain_value.py`'s output).
- **`SharedTrunkPPOPolicy`** (`train_ppo.py:316`): the same interface as `PPOPolicy`, but a single network (`net_cls=SharedTrunkActorCritic` by default, or `SharedTrunkActorCriticDeep`) with a single optimizer — so `value_loss`'s gradient also flows through the trunk shared with the actor (unlike `PPOPolicy`, where `value_net`'s trunk is entirely independent). The same `choose_card`/`end_episode`/`update_batch`/`save`/`load` methods, adapted to call `net.forward_actor`/`net.forward_critic` instead of two separate networks. `load` (`train_ppo.py:457`) automatically detects whether it receives a native checkpoint of this architecture or a raw `CardNetBig` (in which case `net.load_actor_from_cardnetbig` is used, the critic part starting out randomly initialized).
- **`SharedTrunkAuxPPOPolicy(SharedTrunkPPOPolicy)`** (`train_ppo.py:474`): adds the auxiliary task (predicting the number of trumps for the 3 other seats). `choose_card` additionally computes `aux_target` (`other_trump_counts`) and `aux_valid` (`False` in SA/TA, where the "trump slot" notion doesn't make sense) on every decision. `update_batch()` (`train_ppo.py:554`) adds an MSE auxiliary loss (normalized by its own standard deviation, `aux_coef`) computed only over valid decisions, and returns a 5-tuple `(policy_loss, value_loss, aux_loss, entropy, clip_frac)`.
- **`main()`** (`train_ppo.py:633`): parses the arguments, picks the policy class based on `--architecture` (`small`/`big` → `PPOPolicy`; `shared`/`shared_aux`/`shared_deep` → the matching shared-trunk variant), loads `--load`/`--load-value`, then loops over `--episodes`: plays an episode (`run_episode`), computes the counterfactual reward, calls `policy.end_episode(...)`, triggers `policy.update_batch()` every `--batch-episodes`, logs/evaluates/checkpoints periodically. This is the entry point run directly **and** the one launched as a subprocess by `run_growing_pool_ppo.py`.

### 4.7 `pretrain.py`

Supervised pretraining of `CardNet`/`CardNetBig` by imitating `HeuristicPlayer` — the standard starting point before any RL fine-tuning.

- **`RecordingHeuristicPlayer(HeuristicPlayer)`** (`pretrain.py:27`): replays exactly `HeuristicPlayer`'s logic (no duplication); `play_card(...)` (`pretrain.py:36`) encodes the state (`encode_state`) and the legal moves **before** calling `super().play_card(...)`, then records `(state, action_idx, legal_idx)` into `self.dataset`, a list shared between the 4 seats of the same donne.
- `collect_dataset(n_games, seed=None, ablate_points=False)` (`pretrain.py:49`): simulates `n_games` donnes with 4×`RecordingHeuristicPlayer` sharing the same `dataset`, returns the list of collected examples. Called by `main()`.
- `train_supervised(dataset, epochs, lr, batch_size, device, net_cls=CardNet)` (`pretrain.py:66`): a classic training loop — masks illegal moves' logits to `-inf` before the cross-entropy (the same masks as `NeuralPolicy.choose_card` at inference), prints loss/accuracy per epoch. Called by `main()`.
- **`main()`** (`pretrain.py:98`): `--games`/`--epochs`/`--architecture` (`small`=`CardNet`/`big`=`CardNetBig`)/`--ablate-points-so-far`, orchestrates `collect_dataset` → `train_supervised` → `torch.save(net.state_dict(), args.out)`. It's this output `.pt` file (`imit.pt` by default) that `train.py --load`/`train_ppo.py --load` later consume.

### 4.8 `pretrain_value.py`

Supervised pretraining of `ValueNet` (`train_ppo.py`'s critic) — regresses the exact **counterfactual** target PPO actually trains on (not the raw score: an earlier version on the raw score had good offline R² but hurt final PPO performance, see the module docstring and `remarques_rl.md`).

- `collect_dataset(policy, n_games, seed=None)` (`pretrain_value.py:45`): temporarily instruments `policy.choose_card` (a local monkey-patch, restored at the end) to capture `encode_full_state` on every decision, simulates `n_games` donnes with `policy` at seats 0/2 against `HeuristicPlayer` at seats 1/3 (the same scheme as `train.run_episode`), computes `target = reward - counterfactual_reward(hands, dealer)` for every donne (a direct import of `train.counterfactual_reward`), and assigns this single target to every state encountered in the donne. Called by `main()`.
- `train_supervised(dataset, epochs, lr, batch_size, device)` (`pretrain_value.py:87`): a classic MSE regression, prints MSE and R² per epoch. Called by `main()`.
- **`main()`** (`pretrain_value.py:117`): `--policy` (the checkpoint loaded at seats 0/2 for collection, `imit.pt` by default via `PPOPolicy`), `--games`/`--epochs` (notably higher than `pretrain.py` — the residual after subtracting the counterfactual is a much weaker signal), orchestrates `collect_dataset` → `train_supervised` → saving. Output (`value_imit.pt` by default) consumed by `train_ppo.py --load-value`.

### 4.9 `run_growing_pool.py`

REINFORCE orchestrator: splits a training run into successive `train.py` segments, each adding its own checkpoint to the next segment's opponent pool (with PFSP). Contains no training logic itself — only command-line construction and subprocess control.

- `_parse_final_winrates(log_path)` (`run_growing_pool.py:34`): reparses a segment log's last `"PFSP picks total: ..."` line (printed by `PFSPSampler.summary()`) to recover the policy's final win rate against every opponent of that segment. Called by `main()` only if `--pool-max-size` is used.
- `_prune_pool(pool_checkpoints, keep_every)` (`run_growing_pool.py:50`): keeps 1 checkpoint out of every `keep_every` (by segment index), plus always the most recent — reduces the dilution of the PFSP budget per opponent as the pool grows. Called by `main()` at every segment (`--pool-keep-every`).
- **`main()`** (`run_growing_pool.py:67`): for every segment, computes the candidate pool (excludes `--pool-exclude`, applies `_prune_pool`, then prunes by weakness down to `--pool-max-size` if needed using the previous segment's `_parse_final_winrates` — never the very last one added), builds the `python -u train.py --load ... --opponent heuristic,<pool> --pfsp ...` command, runs it via `subprocess.run(..., stdout=logf, stderr=subprocess.STDOUT)` writing to `seg_<N>.log`, stops (`SystemExit`) if the segment fails, otherwise adds the produced checkpoint to the pool and continues. `--start-segment` allows resuming an interrupted orchestration.

### 4.10 `run_growing_pool_ppo.py`

PPO equivalent of 4.9 — the same segmentation/pool logic, drives `train_ppo.py` instead of `train.py`. Simpler: no pool-pruning options (`--pool-keep-every`/`--pool-max-size`/`--pool-exclude` don't exist here, only in the REINFORCE version — this is a first attempt, never yet tested with these refinements).

- **`main()`** (`run_growing_pool_ppo.py:42`): the same structure as `run_growing_pool.py::main` but without the pruning functions — the pool grows without limit. Handles `--load-value`: the first segment starts from `--init-load` (policy only, random critic); the following segments reload both policy **and** critic from the same previous segment file (`--load` and `--load-value` point to the same path, since `PPOPolicy.save()` writes both into a single file) — except for `--architecture shared`/`shared_aux`, where `--load-value` doesn't make sense (a single network) and is never passed.

### 4.11 `eval_policy.py`

Evaluates one or more checkpoints (greedy mode) against `HeuristicPlayer` alone, on exactly the same set of donnes for a fair comparison.

- `_leads_to_all_pass(hands, dealer)` (`eval_policy.py:46`): replays a given donne with 4 `HeuristicPlayer`s (`bid()` has no randomness, so this holds for any checkpoint that inherits this same `bid()` via `RLPlayer`) and detects whether `GameEngine.run_auction()` triggered its internal redeal (`engine.history['deal_hands'] != hands`, cf. `game.py:140-143`). Called by `generate_fixed_deals`.
- **`generate_fixed_deals(n_games, seed)`** (`eval_policy.py:58`): generates `n_games` `(hands, dealer)` pairs **once**, independently of any checkpoint, discarding those that would trigger an internal redeal — essential so `deal(hands=...)` deals exactly the same cards to every checkpoint evaluated. This fixes a real pitfall documented in the module: without hands fixed in advance, as soon as one checkpoint plays a different card from another, `HeuristicPlayer._choose_defausse_suit()` (the only source of randomness during play) consumes a different number of `random` draws, which makes the whole global random stream diverge for the rest of the compared games. Called by `main()` and by `eval_matchup.py::main`.
- `run_fixed_episode(players_factory, hands, dealer)` (`eval_policy.py:78`): plays a donne with fixed hands, returns `team_points[0] - team_points[1]`. Called by `evaluate_fixed`.
- **`evaluate_fixed(players_factory, deals, seed)`** (`eval_policy.py:87`): plays every deal in `deals` with `players_factory`, reseeds `random` beforehand (so the heuristic discard's residual randomness restarts from the same point for every checkpoint compared), returns `(avg, win_rate)`. Called by `main()` and reused directly by `eval_matchup.py`.
- **`main()`** (`eval_policy.py:102`): `checkpoints` (positional list, `'heuristic'` counting as the reference pseudo-checkpoint), `--architecture` (applies to every non-`heuristic` checkpoint of the call — mixing several architectures needs several calls with the same `--seed`/`--games`), generates the fixed donnes once (`generate_fixed_deals`), then for each checkpoint builds a `factory` (`HeuristicPlayer×4` or an `RLPlayer` around a `PPOPolicy`/`SharedTrunkPPOPolicy`/`SharedTrunkAuxPPOPolicy` loaded in greedy mode) and calls `evaluate_fixed`. Prints a table sorted by descending `avg`.

### 4.12 `eval_matchup.py`

Round-robin — each checkpoint against **every other** (not just against `heuristic`), needed to judge a self-play-trained checkpoint's true versatility (its `eval_avg` against `heuristic` alone can be misleading — flat or even down — while it's actually gotten stronger against other playing styles, cf. `rl_experiments_v3/README.md`). Re-imports `generate_fixed_deals`/`evaluate_fixed` from `eval_policy.py` as-is.

- `parse_spec(raw, default_architecture)` (`eval_matchup.py:37`): decodes a CLI argument — `'heuristic'`, or `architecture:path` (an explicit prefix, e.g. `big:seg_300000.pt`), or a bare path that uses `--architecture`. Allows mixing several architectures within a **single** round-robin. Called by `main()`.
- `load_spec(path, architecture='small')` (`eval_matchup.py:50`): `None` for `'heuristic'`, otherwise instantiates the right policy class (`PPOPolicy`/`SharedTrunkPPOPolicy`/`SharedTrunkAuxPPOPolicy` depending on the architecture) and loads it in greedy mode — once per spec, reused for every matchup that involves it. Called by `main()`.
- `make_player(path, policy, name)` (`eval_matchup.py:70`): `HeuristicPlayer` or `RLPlayer(name, policy)` depending on whether `path == 'heuristic'`. Called by `matchup_factory`.
- `matchup_factory(path_a, policy_a, path_b, policy_b)` (`eval_matchup.py:76`): builds a matchup's `factory` — A at seats 0/2, B at seats 1/3 (same convention as `eval_policy.py`). Called by `main()` for every ordered pair `(a, b)` with `a != b`.
- **`main()`** (`eval_matchup.py:88`): parses the `specs`, generates the fixed donnes once, loads every policy once (`load_spec`), then for **every ordered pair** calls `evaluate_fixed` (re-imported from `eval_policy.py`) via `matchup_factory`, and prints both the line-by-line detail and a summary cross table. An implicit methodological note (documented in `CLAUDE.md`, not in this file): `skill(X,Y) = ((X vs Y) − (Y vs X)) / 2` corrects the seat-position bias (~2-5 points favoring seats 0/2) before comparing two checkpoints — this calculation is **not** done by the script itself, do it manually on its output.

### 4.13 `play_test.py`

The simplest script to run an isolated game with no human interaction and keep a JSON trace — the first testing reflex after a change to `game.py`/`players.py`.

- `load_hands(path)` (`play_test.py:8`): loads and validates a `--hands-file` file (a list of 4 lists of card strings). Called by `main()`.
- `build_players(strategies)` (`play_test.py:16`): parses `--strategies` (4 comma-separated values), builds each player via `create_player` (players.py), and attaches a **shared** `SimplePolicy()` to any `RLPlayer` without a policy — so the `'rl'` strategy here is always an untrained random policy, never a real checkpoint (unlike `play_interactive.py`, which knows how to load `rl:architecture:path`). Called by `run_game`.
- `ta_points_to_normal_scale(raw_points)` (`play_test.py:29`): converts a TA score (the 248/15-point scale) to the normal 162/10-point scale, for a display comparable to suit/SA contracts — display logic only, **with no effect on the real scoring** computed by `GameEngine.play()`. Called by `main()` right before the final display.
- **`run_game(strategies, hands, dealer, out_path)`** (`play_test.py:40`): builds the players, instantiates a `CoincheEnv`, calls `reset(hands)` then `step(0)` (the passed action is ignored, cf. §4.3), writes the history to JSON if `out_path` is given. Returns `(contract, team_points, history)`. Called only by `main()`.
- **`main()`** (`play_test.py:54`): parses the CLI arguments, calls `run_game`, prints the final contract and the taking team's points (converted via `ta_points_to_normal_scale` if TA).

### 4.14 `play_interactive.py`

A game played on the keyboard against bots, with live AI advice and an HTML replay generated at the end.

- `load_hands(path)` (`play_interactive.py:12`): identical to `play_test.load_hands` (duplicated, not imported — the two scripts are deliberately independent).
- **`_load_rl_policy(spec)`** (`play_interactive.py:20`): parses `spec = 'architecture:path'` (e.g. `'big:rl_experiments_v4/.../seg_750000.pt'`), instantiates `NeuralPolicy(net_cls=CardNet or CardNetBig)` and **actually loads the checkpoint** in greedy mode (`record=False`) — this is where, unlike `play_test.py`, a real trained checkpoint can be played. Called by `build_one_player`.
- `build_one_player(strat, name)` (`play_interactive.py:40`): if `strat` starts with `'rl:'`, builds an `RLPlayer` around `_load_rl_policy(...)`; otherwise delegates to `create_player` (players.py), with the same `SimplePolicy` fallback as `play_test.py` for an `RLPlayer` without a policy. Called by `build_players` and directly by `main()` to build the `advisor`.
- `build_players(strategies)` (`play_interactive.py:49`): parses `--strategies` (4 values) and calls `build_one_player` for each. Called by `main()`.
- `_with_suffix(path, suffix)` (`play_interactive.py:54`): inserts a suffix before a path's extension (e.g. to generate `game_interactive_botreplay.json` alongside `game_interactive.json`). Called by `main()` for the optional bot replay.
- **`main()`** (`play_interactive.py:59`): determines the human seat (`human_idx`) and the reference bot spec (`bot_spec`, for the default advisor and the replay), builds the players, attaches an `advisor` to the human player if `--advisor != 'none'`, builds a `GameEngine` and runs `deal()` → `run_auction()` → `play()` directly (**not** via `CoincheEnv`, unlike `play_test.py` — this script needs detailed access to `engine.dealer`/`engine.taker_idx`/`engine.contract` for display), saves the JSON, calls `render_history.generate_page` unless `--no-render`, then interactively offers to replay exactly the same donne + the same contract with bots in place of the human (`replay_engine`, copying `contract`/`taker_idx` onto a new engine rather than rerunning an auction that could diverge).

### 4.15 `render_history.py`

Turns a JSON history (`GameEngine.history`, produced by `play_test.py`, `play_interactive.py`, or an ad hoc script) into a standalone HTML page with one tab per trick.

- `card_html(card)` (`render_history.py:18`): a card string (`"AP"`) → a `<span>` colored by suit (`CARD_BG`). Called by `hand_html`/the trick-display loops.
- `sort_hand(cards, trump)` (`render_history.py:27`): sorts a hand for display — by suit (fixed `P/C/K/T` order) then by real strength within each suit (`TRUMP_ORDER` if trump/TA, `NORMAL_ORDER` otherwise). Called by `player_box_html` and by the initial-hands construction in `generate_page`.
- `hand_html(cards)` (`render_history.py:43`): joins a list of `card_html(...)`, or `"(empty)"`. Called by `player_box_html`.
- `card_point(card, trump)` (`render_history.py:49`): a standalone reimplementation (not an import of `GameEngine.card_point` — this module only depends on `game.py`'s constant tables, not on `GameEngine` itself) of a card's point value. Called by `running_team_scores`.
- **`running_team_scores(history, trump)`** (`render_history.py:61`): recomputes, trick by trick, the running score (card points + belote + last-trick bonus), **exactly** replicating `GameEngine.play()`'s logic — including capot detection (every trick won by the same team) and the belote seat (trump King+Queen initially held by the same player, credited as soon as the 2nd of the two cards is played, regardless of who wins the trick). Returns `(running: a list of (score0, score1) per trick, belote_info)`. Called by `generate_page`.
- **`coinche_final_scores(history, raw_final_scores, belote_info)`** (`render_history.py:115`): if the contract is coinched, replaces the raw trick score with the flat award (`160 + 2×contract` for the winner, `0` for the other) — replicates `GameEngine.play()`'s coinche branch. Returns `None` if the contract isn't coinched. Called by `generate_page`.
- `build_stages(history)` (`render_history.py:137`): rebuilds, trick by trick, the state of the hands (removing played cards as it goes) and the list of already-completed tricks — the data structure each HTML panel displays. Called by `generate_page`.
- `player_box_html(seat, css_class, hand_cards, trump, leader_seat=None)` (`render_history.py:157`): a seat's HTML block (name, sorted hand, a "lead" badge if that seat led the trick). Called by `generate_page` (4 times per trick panel).
- **`generate_page(history, out_path)`** (`render_history.py:165`): the main function — assembles `build_stages`, `running_team_scores`, `coinche_final_scores`, the auction and initial-hands blocks, one HTML panel per trick (with belote/lead badges), injects it all into a self-contained HTML/CSS/JS template (clickable tabs in vanilla JS, no external dependency) and writes the result to `out_path`. Called by `main()` **and directly by `play_interactive.py::main`** (importing `render_history.generate_page`) — the only coupling point between the two scripts.
- **`main()`** (`render_history.py:343`): `--history`/`--out`, loads the JSON and calls `generate_page`. The entry point for the `python3 render_history.py --history ... --out ...` command.
