"""Evaluates one or more checkpoints in greedy mode over N games against
HeuristicPlayer (same seats/team as train.py: policy at seats 0/2). Loaded
via PPOPolicy (train_ppo.py), whose `load()` accepts either a CardNet
checkpoint (REINFORCE, train.py) or an ActorCriticNet checkpoint (PPO) -- in
greedy mode (`record=False`) only the policy head matters, the value head
(absent/random on a REINFORCE checkpoint) doesn't come into play.

Unlike an earlier version that reseeded `random` before each checkpoint and
let each game deal freshly (`engine.deal()` with no hands given), THIS
version generates the N (hands, dealer) pairs once -- INDEPENDENTLY of any
checkpoint -- then replays exactly the same hands for each via
`engine.deal(hands=...)`. The earlier version did NOT guarantee identical
hands beyond the very first few games: as soon as one checkpoint plays a
different card from another at some point in the game,
HeuristicPlayer._choose_defausse_suit() (HeuristicPlayer's only source of
randomness during play, outside bidding) can consume a different number of
`random` draws, which makes the whole global `random` state diverge for the
rest of the loop -- verified empirically: hands diverged as early as game 4
out of 100 between two checkpoints with the same starting seed.

Second bug fixed: if the generated hands lead to "everyone passes",
`GameEngine.run_auction()` INTERNALLY redeals with `self.deal()` with NO
argument (game.py:139-143) -- RANDOM hands that silently overwrite the ones
given. `generate_fixed_deals()` filters out such hands at generation time
(checked via a throwaway Heuristic x4 auction, bid() being random-free and
identical to what RLPlayer does since it inherits it as-is) so that the
fixed-hands replay can never fall into this case.

The special pseudo-checkpoint "heuristic" evaluates HeuristicPlayer against
HeuristicPlayer at all 4 seats (no policy loaded): serves as a reference for
what win_rate/point gap results from donne luck alone (dealer, hands) when
both sides play strictly the same strategy.

Usage:
    python eval_policy.py --games 1500 heuristic rl_experiments/imit.pt rl_experiments/exp_lowlr_lowent/final.pt
"""
import argparse
import random

from coinche.game import GameEngine
from coinche.players import HeuristicPlayer, RLPlayer
from coinche.rl_agent import CardNet, CardNetBig, SharedTrunkActorCriticDeep
from train_ppo import PPOPolicy, SharedTrunkPPOPolicy, SharedTrunkAuxPPOPolicy, torch


def _leads_to_all_pass(hands, dealer):
    """True if these hands, auctioned by 4 HeuristicPlayer (bid() is
    random-free, identical for RLPlayer which inherits it as-is -- so this
    test holds for any checkpoint at seats 0/2), make everyone pass and
    trigger game.py:139-143's internal redeal."""
    players = [HeuristicPlayer(f'D{i}') for i in range(4)]
    engine = GameEngine(players, dealer=dealer)
    engine.deal(hands=hands)
    engine.run_auction()
    return engine.history['deal_hands'] != hands


def generate_fixed_deals(n_games, seed):
    """Generates n_games (hands, dealer) pairs once, independent of any
    checkpoint, discarding hands that would trigger an internal redeal (cf.
    the module docstring) -- so `deal(hands=...)` deals exactly the same
    cards to every checkpoint evaluated with this list."""
    random.seed(seed)
    deals = []
    i = 0
    while len(deals) < n_games:
        dealer = i % 4
        dummy = [HeuristicPlayer(f'D{k}') for k in range(4)]
        dummy_engine = GameEngine(dummy, dealer=dealer)
        dummy_engine.deal()
        hands = dummy_engine.history['deal_hands']
        if not _leads_to_all_pass(hands, dealer):
            deals.append((hands, dealer))
        i += 1
    return deals


def run_fixed_episode(players_factory, hands, dealer):
    players = players_factory()
    engine = GameEngine(players, dealer=dealer)
    engine.deal(hands=hands)
    engine.run_auction()
    team_points, _contract = engine.play()
    return team_points[0] - team_points[1]


def evaluate_fixed(players_factory, deals, seed):
    total = 0.0
    wins = 0
    random.seed(seed)  # same starting point for Heuristic's residual randomness (discards)
    for hands, dealer in deals:
        reward = run_fixed_episode(players_factory, hands, dealer)
        total += reward
        if reward > 0:
            wins += 1
    n = len(deals)
    avg = total / n if n else 0.0
    win_rate = wins / n if n else 0.0
    return avg, win_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', help='Paths to the weights to evaluate (or "heuristic").')
    parser.add_argument('--games', type=int, default=1500, help='Number of donnes (shared by all checkpoints).')
    parser.add_argument('--seed', type=int, default=42,
                         help='Seed to generate the fixed donnes (once) and reapplied before each checkpoint.')
    parser.add_argument('--ablate-points-so-far', action='store_true',
                         help="Applies the same ablation flag used at training time (train_ppo.py/"
                              "pretrain.py --ablate-points-so-far) to ALL checkpoints of this call -- "
                              "to compare one ablated group against another, run eval_policy.py once "
                              "per group (same --games/--seed => same generated donnes, cf. generate_fixed_deals).")
    parser.add_argument('--architecture', choices=['small', 'big', 'shared', 'shared_aux', 'shared_deep'],
                         default='small',
                         help="'small' = CardNet (default), 'big' = CardNetBig, 'shared' = "
                              "SharedTrunkActorCritic, 'shared_aux' = SharedTrunkActorCriticAux "
                              "(remarques_rl.md point 6, rl_experiments_v5), 'shared_deep' = "
                              "SharedTrunkActorCriticDeep (3-layer trunk, 1-layer actor head, "
                              "2026-07-28 discussion) -- applies to ALL non-heuristic "
                              "checkpoints of this call; mixing several architectures "
                              "needs a separate call per group (same --games/--seed => same donnes).")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch is required to evaluate a NeuralPolicy (see requirements.txt).")

    deals = generate_fixed_deals(args.games, args.seed)
    policy_net_cls = CardNetBig if args.architecture == 'big' else CardNet

    results = []
    for path in args.checkpoints:
        if path == 'heuristic':
            factory = lambda: [HeuristicPlayer(f'H{i}') for i in range(4)]
        else:
            if args.architecture == 'shared_aux':
                policy = SharedTrunkAuxPPOPolicy()
            elif args.architecture == 'shared_deep':
                policy = SharedTrunkPPOPolicy(net_cls=SharedTrunkActorCriticDeep)
            elif args.architecture == 'shared':
                policy = SharedTrunkPPOPolicy()
            else:
                policy = PPOPolicy(ablate_points=args.ablate_points_so_far, policy_net_cls=policy_net_cls)
            policy.load(path)
            policy.record = False
            factory = lambda policy=policy: [
                RLPlayer('P0', policy), HeuristicPlayer('P1'),
                RLPlayer('P2', policy), HeuristicPlayer('P3'),
            ]
        avg, win_rate = evaluate_fixed(factory, deals, args.seed)
        results.append((path, avg, win_rate))
        print(f"{path:55s}  eval_avg({args.games})={avg:+7.2f}  win_rate={100*win_rate:5.1f}%")

    print()
    print("Summary (same %d donnes, fixed once, for all checkpoints, seed=%d):"
          % (args.games, args.seed))
    for path, avg, win_rate in sorted(results, key=lambda r: -r[1]):
        print(f"  {path:55s}  eval_avg={avg:+7.2f}  win_rate={100*win_rate:5.1f}%")


if __name__ == '__main__':
    main()
