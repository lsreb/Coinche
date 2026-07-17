"""Evalue une ou plusieurs NeuralPolicy en mode glouton sur N parties contre
HeuristicPlayer (memes sieges/equipe que train.py : policy aux sieges 0/2).
Reseed avant chaque checkpoint pour que tous affrontent exactement la meme
sequence de donnes/dealers -- comparaison bas-bruit entre checkpoints, meme
principe que le contre-factuel heuristique de train.py (remarques_rl.md point 3).

Le pseudo-checkpoint special "heuristic" evalue HeuristicPlayer contre
HeuristicPlayer aux 4 sieges (aucune policy chargee) : sert de reference pour
savoir quel win_rate/ecart de points resulte du seul hasard de la donne
(dealer, mains) quand les deux camps jouent strictement la meme strategie.

Usage:
    python eval_policy.py --games 1500 heuristic rl_experiments/imit.pt rl_experiments/exp_lowlr_lowent/final.pt
"""
import argparse
import random

from coinche.game import GameEngine
from coinche.players import HeuristicPlayer
from coinche.rl_agent import NeuralPolicy, torch
from train import evaluate


def run_heuristic_episode(dealer):
    players = [HeuristicPlayer(f'P{i}') for i in range(4)]
    engine = GameEngine(players, dealer=dealer)
    engine.deal()
    engine.run_auction()
    team_points, _contract = engine.play()
    return team_points[0] - team_points[1]


def evaluate_heuristic(n_games, start_dealer=0):
    total = 0.0
    wins = 0
    for i in range(n_games):
        reward = run_heuristic_episode(dealer=(start_dealer + i) % 4)
        total += reward
        if reward > 0:
            wins += 1
    avg = total / n_games if n_games else 0.0
    win_rate = wins / n_games if n_games else 0.0
    return avg, win_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', help='Chemins des poids a evaluer.')
    parser.add_argument('--games', type=int, default=1500, help='Nombre de donnes par checkpoint.')
    parser.add_argument('--seed', type=int, default=42,
                         help='Seed reappliquee avant chaque checkpoint (memes donnes pour tous).')
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch est requis pour evaluer une NeuralPolicy (voir requirements.txt).")

    results = []
    for path in args.checkpoints:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if path == 'heuristic':
            avg, win_rate = evaluate_heuristic(args.games)
        else:
            policy = NeuralPolicy()
            policy.load(path)
            avg, win_rate = evaluate(policy, args.games)
        results.append((path, avg, win_rate))
        print(f"{path:55s}  eval_avg({args.games})={avg:+7.2f}  win_rate={100*win_rate:5.1f}%")

    print()
    print("Resume (memes donnes pour tous les checkpoints, seed=%d) :" % args.seed)
    for path, avg, win_rate in sorted(results, key=lambda r: -r[1]):
        print(f"  {path:55s}  eval_avg={avg:+7.2f}  win_rate={100*win_rate:5.1f}%")


if __name__ == '__main__':
    main()
