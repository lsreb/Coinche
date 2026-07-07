import argparse
import json
from coinche.players import create_player, RLPlayer
from coinche.rl_agent import SimplePolicy
from coinche.env import CoincheEnv


def load_hands(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 4:
        raise ValueError('hands file must contain a list of 4 hands')
    return data


def build_players(strategies):
    parts = strategies.split(',')
    policy = SimplePolicy()
    players = []
    for i in range(4):
        strat = parts[i] if i < len(parts) else 'random'
        p = create_player(strat, f'P{i}')
        if isinstance(p, RLPlayer) and getattr(p, 'policy', None) is None:
            p.policy = policy
        players.append(p)
    return players


def ta_points_to_normal_scale(raw_points):
    """A TA il y a 248 points en tout et pas de der, au lieu de 162 : la correspondance
    120-135-150-165-180-195-210-225-240 <-> 80-90-100-110-120-130-140-150-160
    (regles_coinche.md §5) donne 15 points TA pour 10 points normaux. Le capot (250)
    n'est pas concerne par cette conversion."""
    if raw_points >= 250:
        return raw_points
    return round(80 + (raw_points - 120) * (10 / 15))


def run_game(strategies, hands, dealer, out_path):
    players = build_players(strategies)
    env = CoincheEnv(players=players, dealer=dealer, agent_seat=0)
    obs, info = env.reset(hands)
    _, _, _, info2 = env.step(0)
    history = info2.get('history', info.get('history'))
    contract = info2['contract']
    team_points = info2['team_points']
    if out_path:
        with open(out_path, 'w') as f:
            json.dump(history, f, indent=2)
    return contract, team_points, history


def main():
    parser = argparse.ArgumentParser(description='Run isolated Coinche test games.')
    parser.add_argument('--strategies', default='heuristic,random,heuristic,random',
                        help='Comma-separated strategies for seats 0-3. Options: random, heuristic, rl, user')
    parser.add_argument('--hands-file', default=None,
                        help='JSON file with 4 hands, each hand as list of card strings like ["AP","10P",...].')
    parser.add_argument('--dealer', type=int, default=0,
                        help='Dealer seat index (0-3) for the game.')
    parser.add_argument('--out', default='game_history.json',
                        help='Output JSON path for the game history.')
    args = parser.parse_args()

    hands = None
    if args.hands_file:
        hands = load_hands(args.hands_file)

    contract, team_points, history = run_game(args.strategies, hands, args.dealer, args.out)
    taker_team = history['contract']['taker'] % 2
    level = 'capot' if contract.capot else contract.level
    points = team_points[taker_team]
    if contract.trump == 'TA':
        points = ta_points_to_normal_scale(points)
    print(f"Contrat demande: {level}{contract.trump} (preneur: seat {history['contract']['taker']})")
    print('Points marques par l\'equipe preneuse:', points)
    print('history saved to', args.out)


if __name__ == '__main__':
    main()
