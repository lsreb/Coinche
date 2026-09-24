import argparse
import json
import os
import random

from coinche.game import GameEngine
from coinche.players import create_player, RLPlayer, HumanPlayer, SEAT_LABELS
from coinche.rl_agent import SimplePolicy
from render_history import generate_page


def load_hands(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 4:
        raise ValueError('hands file must contain a list of 4 hands')
    return data


def _load_rl_policy(spec):
    """spec = 'architecture:path' (e.g. 'big:rl_experiments_v4/exp_growing_pool_bignet/seg_300000.pt'),
    same convention as eval_matchup.py. Loads a REINFORCE checkpoint (NeuralPolicy) in greedy
    mode (record=False), not create_player('rl')'s default random policy."""
    from coinche.rl_agent import torch, CardNet, CardNetBig, NeuralPolicy
    if torch is None:
        raise SystemExit("torch is required to load an RL checkpoint (see requirements.txt).")
    if ':' not in spec:
        raise SystemExit(f"Expected format 'architecture:path' for an rl: bot (got {spec!r}).")
    arch, path = spec.split(':', 1)
    architectures = {'small': CardNet, 'big': CardNetBig}
    net_cls = architectures.get(arch)
    if net_cls is None:
        raise SystemExit(f"Unknown RL architecture {arch!r} (options: {sorted(architectures)}).")
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
    parser = argparse.ArgumentParser(description='Play a Coinche donne interactively against bots.')
    parser.add_argument('--strategies', default='heuristic,heuristic,human,heuristic',
                         help="Strategies for seats 0-3 (N,W,S,E), comma-separated. "
                              "Options: random, heuristic, human, rl (random policy), or "
                              "rl:architecture:path (e.g. rl:big:rl_experiments_v4/exp_growing_pool_bignet/"
                              "seg_300000.pt) to load a real trained checkpoint. By default, only "
                              "seat S (2nd after N) is human.")
    parser.add_argument('--hands-file', default=None,
                         help='JSON with 4 hands, e.g. ["AP","10P",...]. Ignored on a redeal (everyone passes).')
    parser.add_argument('--dealer', type=int, default=None,
                         help='Dealer seat (0-3). Random on each run by default.')
    parser.add_argument('--out', default='game_interactive.json', help="Path to save the JSON history.")
    parser.add_argument('--no-render', action='store_true', help='Do not generate the HTML replay page.')
    parser.add_argument('--render-out', default='game_interactive_viewer.html')
    parser.add_argument('--advisor', default='auto',
                         help="Bot whose suggestion is shown live on each human decision "
                              "(same format as --strategies: heuristic, random, rl:architecture:path). "
                              "'auto' (default) reuses the first bot seat's strategy. 'none' disables it.")
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

    dealer = args.dealer if args.dealer is not None else random.randint(0, 3)
    print(f"Dealer: {SEAT_LABELS[dealer]}  (first to speak: {SEAT_LABELS[(dealer + 1) % 4]})")
    engine = GameEngine(players, dealer=dealer)
    engine.deal(hands)
    engine.run_auction()
    team_points, contract = engine.play()
    for p in players:
        if isinstance(p, HumanPlayer):
            p.show_new_tricks()  # catch up on the last trick, never seen if we weren't last to play in it

    taker_team = engine.taker_idx % 2
    level = 'capot' if contract.capot else contract.level
    print()
    print(f"Final contract: {level}{contract.trump}" + (' (coinched)' if contract.coinched else ''))
    print(f"Taking team points: {team_points[taker_team]}  |  Defense points: {team_points[1 - taker_team]}")

    with open(args.out, 'w') as f:
        json.dump(engine.history, f, indent=2)
    print('History saved to', args.out)

    if not args.no_render:
        generate_page(engine.history, args.render_out)
        print('Replay generated at', args.render_out)

    if human_idx is not None:
        answer = input(f"\nGenerate the same deal + contract played by 4x {bot_spec!r}? (y/N): ").strip().lower()
        if answer in ('y', 'yes'):
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
            print(f"Taking team points (bots): {replay_points[taker_team]}  |  "
                  f"Defense points (bots): {replay_points[1 - taker_team]}")

            replay_out = _with_suffix(args.out, '_botreplay')
            with open(replay_out, 'w') as f:
                json.dump(replay_engine.history, f, indent=2)
            print('Replay history saved to', replay_out)

            if not args.no_render:
                replay_render_out = _with_suffix(args.render_out, '_botreplay')
                generate_page(replay_engine.history, replay_render_out)
                print('Replay page generated at', replay_render_out)


if __name__ == '__main__':
    main()
