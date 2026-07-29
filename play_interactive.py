import argparse
import json
import os

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


def build_one_player(strat, name):
    if strat.startswith('rl:'):
        return RLPlayer(name, _load_rl_policy(strat[len('rl:'):]))
    p = create_player(strat, name)
    if isinstance(p, RLPlayer) and getattr(p, 'policy', None) is None:
        p.policy = SimplePolicy()
    return p


def build_players(strategies):
    parts = strategies.split(',')
    return [build_one_player(parts[i] if i < len(parts) else 'heuristic', f'P{i}') for i in range(4)]


def _with_suffix(path, suffix):
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


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
    parser.add_argument('--advisor', default='auto',
                         help="Bot dont la suggestion s'affiche en temps reel a chaque decision humaine "
                              "(meme format que --strategies : heuristic, random, rl:architecture:chemin). "
                              "'auto' (defaut) reprend la strategie du premier siege bot. 'none' desactive.")
    args = parser.parse_args()

    strat_parts = args.strategies.split(',')
    strat_parts += ['heuristic'] * (4 - len(strat_parts))
    human_idx = next((i for i, s in enumerate(strat_parts) if s == 'human'), None)
    bot_spec = next((s for i, s in enumerate(strat_parts) if i != human_idx and s != 'human'), 'heuristic')

    hands = load_hands(args.hands_file) if args.hands_file else None
    players = build_players(args.strategies)
    if human_idx is not None and args.advisor != 'none':
        advisor_spec = bot_spec if args.advisor == 'auto' else args.advisor
        players[human_idx].advisor = build_one_player(advisor_spec, 'IA')

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

    if human_idx is not None:
        answer = input(f"\nGénérer la même donne + contrat jouée par 4x {bot_spec!r} ? (y/N) : ").strip().lower()
        if answer in ('y', 'yes', 'o', 'oui'):
            replay_players = [build_one_player(bot_spec, f'B{i}') for i in range(4)]
            replay_engine = GameEngine(replay_players, dealer=engine.dealer)
            replay_engine.deal(engine.history['deal_hands'])
            replay_engine.contract = engine.contract
            replay_engine.taker_idx = engine.taker_idx
            replay_engine.history['contract'] = {
                'level': engine.contract.level,
                'trump': engine.contract.trump,
                'taker': engine.taker_idx,
                'capot': engine.contract.capot,
                'coinched': engine.contract.coinched,
            }
            replay_points, _ = replay_engine.play()
            print(f"Points équipe preneuse (bots) : {replay_points[taker_team]}  |  "
                  f"Points défense (bots) : {replay_points[1 - taker_team]}")

            replay_out = _with_suffix(args.out, '_botreplay')
            with open(replay_out, 'w') as f:
                json.dump(replay_engine.history, f, indent=2)
            print('Historique du replay sauvegardé dans', replay_out)

            if not args.no_render:
                replay_render_out = _with_suffix(args.render_out, '_botreplay')
                generate_page(replay_engine.history, replay_render_out)
                print('Relecture du replay générée dans', replay_render_out)


if __name__ == '__main__':
    main()
