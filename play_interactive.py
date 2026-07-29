import argparse
import json

from coinche.game import GameEngine
from coinche.players import create_player, RLPlayer
from coinche.rl_agent import SimplePolicy
from render_history import generate_page


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
        strat = parts[i] if i < len(parts) else 'heuristic'
        p = create_player(strat, f'P{i}')
        if isinstance(p, RLPlayer) and getattr(p, 'policy', None) is None:
            p.policy = policy
        players.append(p)
    return players


def main():
    parser = argparse.ArgumentParser(description='Joue une donne de Coinche en interactif contre des bots.')
    parser.add_argument('--strategies', default='heuristic,heuristic,human,heuristic',
                         help="Strategies des sieges 0-3 (N,W,S,E), separees par des virgules. "
                              "Options : random, heuristic, rl, human. Par defaut, seul le siege S (2e apres N) "
                              "est humain.")
    parser.add_argument('--hands-file', default=None,
                         help='JSON avec 4 mains, ex ["AP","10P",...]. Ignore en cas de redonne (tout le monde passe).')
    parser.add_argument('--dealer', type=int, default=0, help='Siege du donneur (0-3).')
    parser.add_argument('--out', default='game_interactive.json', help="Chemin de sauvegarde de l'historique JSON.")
    parser.add_argument('--no-render', action='store_true', help='Ne pas generer la page HTML de relecture.')
    parser.add_argument('--render-out', default='game_interactive_viewer.html')
    args = parser.parse_args()

    hands = load_hands(args.hands_file) if args.hands_file else None
    players = build_players(args.strategies)
    engine = GameEngine(players, dealer=args.dealer)
    engine.deal(hands)
    engine.run_auction()
    team_points, contract = engine.play()

    taker_team = engine.taker_idx % 2
    level = 'capot' if contract.capot else contract.level
    print()
    print(f"Contrat final : {level}{contract.trump}" + (' (coinché)' if contract.coinched else ''))
    print(f"Points équipe preneuse : {team_points[taker_team]}  |  Points défense : {team_points[1 - taker_team]}")

    with open(args.out, 'w') as f:
        json.dump(engine.history, f, indent=2)
    print('Historique sauvegardé dans', args.out)

    if not args.no_render:
        generate_page(engine.history, args.render_out)
        print('Relecture générée dans', args.render_out)


if __name__ == '__main__':
    main()
