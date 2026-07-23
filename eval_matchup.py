"""Round-robin policy-vs-policy, contrairement a eval_policy.py qui ne compare
qu'un checkpoint a la fois contre HeuristicPlayer. Sert a verifier si un
checkpoint entraine en pool grandissant (self-play, run_growing_pool.py) est
plus polyvalent qu'un checkpoint entraine seulement contre heuristic, meme si
son eval_avg contre heuristic seul est plus bas (cf.
rl_experiments_v3/README.md, section pool grandissant) -- eval_avg contre
heuristic seul ne mesure que l'exploitation d'un adversaire fixe, pas la
polyvalence face a d'autres styles de jeu.

Reutilise generate_fixed_deals/evaluate_fixed de eval_policy.py (memes N
donnes fixees pour tous les appariements, memes pieges deja corriges :
redonne interne sur "tout le monde passe", divergence du random global).

Chaque paire ordonnee (A, B) avec A != B est jouee : A aux sieges 0/2, B aux
sieges 1/3, sur les memes N donnes. "heuristic" est un spec valide comme dans
eval_policy.py.

Usage:
    python eval_matchup.py --games 3000 heuristic ckpt1.pt ckpt2.pt ckpt3.pt
"""
import argparse

from coinche.players import HeuristicPlayer, RLPlayer
from eval_policy import generate_fixed_deals, evaluate_fixed
from train_ppo import PPOPolicy, torch


def load_spec(spec):
    """None pour 'heuristic' (pas de policy a charger), sinon une PPOPolicy
    chargee et mise en mode glouton -- chargee une seule fois par spec, reutilisee
    pour tous les appariements qui l'impliquent."""
    if spec == 'heuristic':
        return None
    policy = PPOPolicy()
    policy.load(spec)
    policy.record = False
    return policy


def make_player(spec, policy, name):
    if spec == 'heuristic':
        return HeuristicPlayer(name)
    return RLPlayer(name, policy)


def matchup_factory(spec_a, policy_a, spec_b, policy_b):
    """A aux sieges 0/2, B aux sieges 1/3 -- meme convention que eval_policy.py
    (une seule policy par equipe, meme objet reutilise aux 2 sieges de l'equipe,
    comme RLPlayer le permet deja partout ailleurs dans ce depot)."""
    def factory():
        return [
            make_player(spec_a, policy_a, 'A0'), make_player(spec_b, policy_b, 'B1'),
            make_player(spec_a, policy_a, 'A2'), make_player(spec_b, policy_b, 'B3'),
        ]
    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specs', nargs='+', help='Checkpoints (ou "heuristic") a opposer deux a deux.')
    parser.add_argument('--games', type=int, default=3000, help='Nombre de donnes par appariement.')
    parser.add_argument('--seed', type=int, default=42,
                         help='Seed pour generer les donnes fixes (une fois, partagees par tous les appariements).')
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch est requis pour evaluer une NeuralPolicy (voir requirements.txt).")

    deals = generate_fixed_deals(args.games, args.seed)
    policies = {spec: load_spec(spec) for spec in args.specs}

    results = {}
    for a in args.specs:
        for b in args.specs:
            if a == b:
                continue
            factory = matchup_factory(a, policies[a], b, policies[b])
            avg, win_rate = evaluate_fixed(factory, deals, args.seed)
            results[(a, b)] = (avg, win_rate)
            print(f"{a:55s} vs {b:55s}  avg={avg:+7.2f}  win_rate={100*win_rate:5.1f}%")

    print()
    print(f"Resume round-robin (memes {args.games} donnes fixees pour tous les appariements, seed={args.seed}) :")
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
