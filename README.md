Coinche (Belote coinchée) minimal RL environment

Structure
- coinche/: core game, players, environment and simple RL policy

Usage
1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Use the `coinche.env.CoincheEnv` to run self-play and plug a policy from `coinche.rl_agent`.

python3 render_history.py --history game_history_test.json --out game_history_viewer.html génère une représentation graphique à partir d'un historique
