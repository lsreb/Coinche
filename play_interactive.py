import argparse
import json

from coinche.game import GameEngine
from coinche.players import create_player, RLPlayer, HumanPlayer
from coinche.rl_agent import SimplePolicy
from render_history import generate_page


def load_hands(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 4:
        raise ValueError('hands file must contain a list of 4 hands')
    return data


def _load_rl_policy(spec):
    """spec = 'architecture:chemin' (ex. 'big:rl_experiments_v4/exp_growing_pool_bignet/seg_300000.pt'),
    meme convention que eval_matchup.py. Charge un checkpoint REINFORCE (NeuralPolicy) en mode
    glouton (record=False), pas la policy aleatoire par defaut de create_player('rl')."""
    from coinche.rl_agent import torch, CardNet, CardNetBig, NeuralPolicy
    if torch is None:
        raise SystemExit("torch est requis pour charger un checkpoint RL (voir requirements.txt).")
    if ':' not in spec:
        raise SystemExit(f"Format attendu 'architecture:chemin' pour un bot rl: (recu {spec!r}).")
    arch, path = spec.split(':', 1)
    architectures = {'small': CardNet, 'big': CardNetBig}
    net_cls = architectures.get(arch)
    if net_cls is None:
        raise SystemExit(f"Architecture RL inconnue {arch!r} (options: {sorted(architectures)}).")
    policy = NeuralPolicy(net_cls=net_cls)
    policy.load(path)
    policy.record = False
    return policy


def build_players(strategies):
    parts = strategies.split(',')
    default_policy = SimplePolicy()
    players = []
    for i in range(4):
        strat = parts[i] if i < len(parts) else 'heuristic'
        if strat.startswith('rl:'):
            p = RLPlayer(f'P{i}', _load_rl_policy(strat[len('rl:'):]))
        else:
            p = create_player(strat, f'P{i}')
            if isinstance(p, RLPlayer) and getattr(p, 'policy', None) is None:
                p.policy = default_policy
        players.append(p)
    return players


def main():
    parser = argparse.ArgumentParser(description='Joue une donne de Coinche en interactif contre des bots.')
    parser.add_argument('--strategies', default='heuristic,heuristic,human,heuristic',
                         help="Strategies des sieges 0-3 (N,W,S,E), separees par des virgules. "
                              "Options : random, heuristic, human, rl (policy aleatoire), ou "
                              "rl:architecture:chemin (ex. rl:big:rl_experiments_v4/exp_growing_pool_bignet/"
                              "seg_300000.pt) pour charger un vrai checkpoint entraine. Par defaut, seul le "
                              "siege S (2e apres N) est humain.")
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
    for p in players:
        if isinstance(p, HumanPlayer):
            p.show_new_tricks()  # rattrape le dernier pli, jamais vu si on n'y jouait pas en dernier

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
