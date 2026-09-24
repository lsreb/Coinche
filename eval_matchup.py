"""Round-robin policy-vs-policy, unlike eval_policy.py which only compares
one checkpoint at a time against HeuristicPlayer. Used to check whether a
checkpoint trained in a growing pool (self-play, run_growing_pool.py) is
more versatile than a checkpoint trained only against heuristic, even if its
eval_avg against heuristic alone is lower (cf. rl_experiments_v3/README.md,
growing-pool section) -- eval_avg against heuristic alone only measures
exploitation of a fixed opponent, not versatility against other playing styles.

Reuses generate_fixed_deals/evaluate_fixed from eval_policy.py (same N fixed
donnes for every matchup, same already-fixed pitfalls: internal redeal on
"everyone passes", global random divergence).

Every ordered pair (A, B) with A != B is played: A at seats 0/2, B at seats
1/3, on the same N donnes. "heuristic" is a valid spec, as in eval_policy.py.

A spec can also be written as `architecture:path` (e.g. `big:seg_300000.pt`,
`shared:seg_300000.pt`) to mix several architectures in the SAME
round-robin (--architecture then only serves as the default value for specs
without an explicit prefix) -- needed to compare two checkpoints of
different architectures head-to-head directly (e.g. growing-pool CardNetBig
vs growing-pool shared trunk).

Usage:
    python eval_matchup.py --games 3000 heuristic ckpt1.pt ckpt2.pt ckpt3.pt
    python eval_matchup.py --games 3000 heuristic big:bignet.pt shared:shared.pt
"""
import argparse

from coinche.players import HeuristicPlayer, RLPlayer
from coinche.rl_agent import CardNet, CardNetBig, SharedTrunkActorCriticDeep
from eval_policy import generate_fixed_deals, evaluate_fixed
from train_ppo import PPOPolicy, SharedTrunkPPOPolicy, SharedTrunkAuxPPOPolicy, torch

ARCHITECTURES = ('small', 'big', 'shared', 'shared_aux', 'shared_deep')


def parse_spec(raw, default_architecture):
    """A CLI spec is either 'heuristic', or `architecture:path` (explicit
    prefix), or a bare path that uses `default_architecture`
    (--architecture) -- returns (architecture, path_or_'heuristic')."""
    if raw == 'heuristic':
        return default_architecture, 'heuristic'
    if ':' in raw:
        prefix, rest = raw.split(':', 1)
        if prefix in ARCHITECTURES:
            return prefix, rest
    return default_architecture, raw


def load_spec(path, architecture='small'):
    """None for 'heuristic' (no policy to load), otherwise a Policy loaded
    and set to greedy mode -- loaded once per spec, reused for every
    matchup that involves it."""
    if path == 'heuristic':
        return None
    if architecture == 'shared_aux':
        policy = SharedTrunkAuxPPOPolicy()
    elif architecture == 'shared_deep':
        policy = SharedTrunkPPOPolicy(net_cls=SharedTrunkActorCriticDeep)
    elif architecture == 'shared':
        policy = SharedTrunkPPOPolicy()
    else:
        policy_net_cls = CardNetBig if architecture == 'big' else CardNet
        policy = PPOPolicy(policy_net_cls=policy_net_cls)
    policy.load(path)
    policy.record = False
    return policy


def make_player(path, policy, name):
    if path == 'heuristic':
        return HeuristicPlayer(name)
    return RLPlayer(name, policy)


def matchup_factory(path_a, policy_a, path_b, policy_b):
    """A at seats 0/2, B at seats 1/3 -- same convention as eval_policy.py
    (a single policy per team, the same object reused at both of the
    team's seats, as RLPlayer already allows everywhere else in this repo)."""
    def factory():
        return [
            make_player(path_a, policy_a, 'A0'), make_player(path_b, policy_b, 'B1'),
            make_player(path_a, policy_a, 'A2'), make_player(path_b, policy_b, 'B3'),
        ]
    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specs', nargs='+', help='Checkpoints (or "heuristic") to pit against each other.')
    parser.add_argument('--games', type=int, default=3000, help='Number of donnes per matchup.')
    parser.add_argument('--seed', type=int, default=42,
                         help='Seed to generate the fixed donnes (once, shared by all matchups).')
    parser.add_argument('--architecture', choices=ARCHITECTURES, default='small',
                         help="'small' = CardNet (default), 'big' = CardNetBig, 'shared' = "
                              "SharedTrunkActorCritic, 'shared_aux' = SharedTrunkActorCriticAux, "
                              "'shared_deep' = SharedTrunkActorCriticDeep -- default architecture "
                              "for specs without an explicit `architecture:path` prefix (see "
                              "the module docstring to mix several architectures).")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch is required to evaluate a NeuralPolicy (see requirements.txt).")

    deals = generate_fixed_deals(args.games, args.seed)
    # key = raw CLI spec (display/dict), value = (architecture, real path or 'heuristic').
    parsed = {raw: parse_spec(raw, args.architecture) for raw in args.specs}
    policies = {raw: load_spec(path, architecture=arch) for raw, (arch, path) in parsed.items()}

    results = {}
    for a in args.specs:
        for b in args.specs:
            if a == b:
                continue
            _, path_a = parsed[a]
            _, path_b = parsed[b]
            factory = matchup_factory(path_a, policies[a], path_b, policies[b])
            avg, win_rate = evaluate_fixed(factory, deals, args.seed)
            results[(a, b)] = (avg, win_rate)
            print(f"{a:55s} vs {b:55s}  avg={avg:+7.2f}  win_rate={100*win_rate:5.1f}%")

    print()
    print(f"Round-robin summary (same {args.games} donnes fixed for all matchups, seed={args.seed}):")
    print(f"{'':55s}", end='')
    for b in args.specs:
        print(f"{b[-20:]:>22s}", end='')
    print()
    for a in args.specs:
        print(f"{a:55s}", end='')
        for b in args.specs:
            if a == b:
                print(f"{'--':>22s}", end='')
            else:
                avg, _ = results[(a, b)]
                print(f"{avg:>+22.2f}", end='')
        print()


if __name__ == '__main__':
    main()
