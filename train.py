"""Entraine par REINFORCE une policy de jeu de la carte pour une equipe entiere
(memes poids sur les deux sieges partenaires), face a une equipe HeuristicPlayer.
Les encheres restent geree par HeuristicPlayer des deux cotes (RLPlayer en herite) :
seul le choix de la carte a jouer, une fois le contrat fixe, est appris.

Usage:
    python train.py --episodes 5000 --eval-every 200
    python train.py --episodes 2000 --load poids.pt --save poids.pt
"""
import argparse
import random
import time

from coinche.game import GameEngine
from coinche.players import RLPlayer, HeuristicPlayer
from coinche.rl_agent import NeuralPolicy, torch

# L'equipe RL controle les sieges 0 et 2 (partenaires) ; 1 et 3 restent heuristiques.
RL_SEATS = (0, 2)


def run_episode(policy, dealer):
    players = [
        RLPlayer('P0', policy), HeuristicPlayer('P1'),
        RLPlayer('P2', policy), HeuristicPlayer('P3'),
    ]
    engine = GameEngine(players, dealer=dealer)
    engine.deal()
    engine.run_auction()
    team_points, _contract = engine.play()
    return team_points[0] - team_points[1]  # equipe RL = siege 0/2, parite 0


def evaluate(policy, n_games, start_dealer=0):
    was_recording = policy.record
    policy.record = False
    total = 0.0
    for i in range(n_games):
        total += run_episode(policy, dealer=(start_dealer + i) % 4)
    policy.record = was_recording
    return total / n_games if n_games else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=2000, help='Nombre de donnes d\'entrainement.')
    parser.add_argument('--eval-every', type=int, default=200, help='Frequence (en episodes) des evaluations gloutonnes.')
    parser.add_argument('--eval-games', type=int, default=100, help='Nombre de donnes par evaluation.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--entropy-beta', type=float, default=0.01,
                         help="Poids du bonus d'entropie dans la loss REINFORCE (force l'exploration, "
                              "utile en particulier apres un --load d'une policy pre-entrainee par imitation).")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save', default=None, help="Chemin pour sauvegarder les poids en fin d'entrainement.")
    parser.add_argument('--load', default=None, help='Chemin pour reprendre depuis des poids sauvegardes.')
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch est requis pour entrainer une NeuralPolicy (voir requirements.txt).")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    policy = NeuralPolicy(lr=args.lr, entropy_beta=args.entropy_beta)
    if args.load:
        policy.load(args.load)
        print('poids charges depuis', args.load)

    window = []
    start = time.perf_counter()
    for ep in range(1, args.episodes + 1):
        reward = run_episode(policy, dealer=ep % 4)
        policy.update(reward)
        window.append(reward)
        if len(window) > 200:
            window.pop(0)

        if ep % args.eval_every == 0:
            avg_train = sum(window) / len(window)
            avg_eval = evaluate(policy, args.eval_games, start_dealer=ep)
            elapsed = time.perf_counter() - start
            print(f"episode {ep:6d}  train_avg={avg_train:+7.1f}  eval_avg({args.eval_games})={avg_eval:+7.1f}  "
                  f"baseline={policy.baseline:+7.1f}  ({elapsed:.1f}s)")

    if args.save:
        policy.save(args.save)
        print('poids sauvegardes dans', args.save)


if __name__ == '__main__':
    main()
