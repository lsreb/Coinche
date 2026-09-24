# Coinche RL

A Python implementation of Coinche (belote coinchée) — full bidding and trick-play rules engine, a hand-written heuristic bot, and a reinforcement-learning pipeline (REINFORCE and PPO) that trains a neural network to play the cards against it.

There's no packaged CLI or web app here: this is a research codebase you run as scripts, built for iterating on RL ideas against a hand-coded opponent baseline.

## What's in here

- **Game engine** (`coinche/game.py`, `coinche/players.py`, `coinche/env.py`) — the rules (`regles_coinche.md`) and a rule-based bot (`HeuristicPlayer`, following `heuristiques.md`). No dependencies beyond the standard library.
- **RL layer** (`coinche/rl_agent.py`, `train.py`, `train_ppo.py`, `pretrain*.py`, `run_growing_pool*.py`, `eval_*.py`) — state encoding, network architectures, and the REINFORCE/PPO trainers. Needs `torch`; degrades gracefully without it.
- **Play & inspect** (`play_test.py`, `play_interactive.py`, `render_history.py`) — run an isolated game, play interactively against bots with a live AI hint, and render any saved game history as a browsable HTML replay.

## Quickstart

```bash
pip install -r requirements.txt   # numpy + torch (torch is optional)

# Sanity-check the game engine and heuristic bidder
python -m coinche.players

# Play a full game between bots and save its history
python play_test.py --strategies heuristic,heuristic,heuristic,heuristic --out game_test.json

# Turn that history into a browsable HTML replay
python3 render_history.py --history game_test.json --out game_history_viewer.html

# Play a hand yourself against bots, with a live AI hint on your turn
python play_interactive.py --strategies heuristic,heuristic,human,heuristic
```

From there, imitation-pretraining a network on the heuristic bot (`pretrain.py`) and fine-tuning it with REINFORCE or PPO (`train.py` / `train_ppo.py`) is the standard next step — see below for where that's documented.

## Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** — a detailed walkthrough: what every script does, the order you'd actually run them in, and a function-by-function reference of the whole codebase (who calls what). Start here if you're new to the repo.
- **`CLAUDE.md`** — a compact command/architecture reference, kept up to date as the project evolves.
- **`regles_coinche.md`** / **`heuristiques.md`** — the rules of Coinche and the specific heuristics `HeuristicPlayer` implements (source of truth for game-logic correctness).
- **`coinche/remarques_rl.md`** and the `rl_experiments*/README.md` files — the running research log of what's been tried in the RL experimentation, what worked, and what didn't. Worth reading before starting new RL work to avoid retreading settled ground.

## Status

There's no automated test suite (no `pytest`); correctness is checked by running games end-to-end and inspecting the JSON/HTML output, plus the round-robin evaluation scripts (`eval_policy.py`, `eval_matchup.py`) for comparing trained checkpoints. The one RL idea confirmed to reliably beat a checkpoint trained only against the heuristic bot is growing-pool self-play (`run_growing_pool.py` / `run_growing_pool_ppo.py`) — see `coinche/remarques_rl.md` for the full experimental history.
