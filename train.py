"""Minimal training script example using the provided environment and SimplePolicy."""
from coinche.env import CoincheEnv
import argparse
from coinche.rl_agent import SimplePolicy
from coinche.players import create_player, RLPlayer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategies', type=str, default='heuristic,random,random,random',
                        help='Comma-separated strategies for seats 0-3. Options: random, user, heuristic, rl')
    args = parser.parse_args()
    parts = args.strategies.split(',')
    # prepare optional policy for RL players
    policy = SimplePolicy()
    players = []
    for i in range(4):
        strat = parts[i] if i < len(parts) else 'random'
        p = create_player(strat, f'P{i}')
        # if RL player and no policy assigned, attach simple policy
        if isinstance(p, RLPlayer) and getattr(p, 'policy', None) is None:
            p.policy = policy
        players.append(p)
    env = CoincheEnv(players=players, dealer=0, agent_seat=0)
    obs = env.reset()
    obs, reward, done, info = env.step(0)
    print('Reward (team points):', reward)

if __name__ == '__main__':
    main()
