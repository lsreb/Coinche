"""Pretrains a ValueNet (train_ppo.py) by supervised regression on the
COUNTERFACTUAL return -- the same quantity train_ppo.py actually trains on
(cf. `train.counterfactual_reward`, remarques_rl.md point 3), NOT the raw
return -- before PPO fine-tuning (train_ppo.py --load-value).

An earlier version of this script regressed on the RAW return
(team_points[0]-team_points[1]), collected on HeuristicPlayer x4 donnes:
good offline R2 (0.28), but worse final performance AFTER PPO fine-tuning
than a randomly initialized ValueNet (verified empirically). Cause:
train_ppo.py only trains value_loss (and the advantage) on
`reward - counterfactual_reward(...)`, the gap to what a heuristic would
have done on the SAME cards -- and this counterfactual absorbs almost all of
the raw return's `is_attacker` signal (verified: an attack/defense gap of
about 190 points on the raw return, only ~17 points on the counterfactual
return, on the same donnes). The ValueNet pretrained on the raw return
therefore started out with confident but systematically biased predictions
for the TRUE PPO target (advantage artificially too negative on attack, too
positive on defense) -- worse than a neutral start.

So this simulates donnes with `--policy` (imit.pt by default, even though
not yet PPO fine-tuned) at seats 0/2 against HeuristicPlayer at seats 1/3 --
exactly like `train.run_episode` -- computes the same counterfactual as
train_ppo.py on the same hands, and regresses ValueNet on that target.

Usage:
    python pretrain_value.py --games 5000 --epochs 20 --out value_imit.pt
    python train_ppo.py --load rl_experiments/imit.pt --load-value value_imit.pt --episodes 50000
"""
import argparse
import random
import time

from coinche.game import GameEngine
from coinche.players import HeuristicPlayer, RLPlayer
from coinche.rl_agent import encode_full_state, torch
from train import counterfactual_reward
from train_ppo import PPOPolicy, ValueNet

try:
    import torch.nn.functional as F
except Exception:
    F = None


def collect_dataset(policy, n_games, seed=None):
    """Simulates n_games donnes, `policy` at seats 0/2 (like
    train.run_episode) against HeuristicPlayer at seats 1/3, and assigns to
    every state encountered at seats 0/2 the counterfactual return of the
    whole donne (same convention as train_ppo.py: a single scalar for all
    of the donne's states, terminal reward, cf. the module docstring)."""
    if seed is not None:
        random.seed(seed)
    dataset = []
    current = []

    orig_choose_card = policy.choose_card
    def instrumented(player, legal, leader, trick, trump):
        # encode_full_state (not encode_state): ValueNet is now a
        # centralized critic (cf. train_ppo.py), same convention as choose_card.
        current.append(encode_full_state(player, trick, trump))
        return orig_choose_card(player, legal, leader, trick, trump)
    policy.choose_card = instrumented
    policy.record = False  # states representative of the policy's best response, not noisy exploration

    start = time.perf_counter()
    for i in range(n_games):
        current.clear()
        dealer = i % 4
        players = [RLPlayer('P0', policy), HeuristicPlayer('P1'),
                   RLPlayer('P2', policy), HeuristicPlayer('P3')]
        engine = GameEngine(players, dealer=dealer)
        engine.deal()
        engine.run_auction()
        team_points, _contract = engine.play()
        reward = team_points[0] - team_points[1]
        hands = engine.history['deal_hands']
        target = reward - counterfactual_reward(hands, dealer=dealer)
        dataset.extend((state, target) for state in current)

    policy.choose_card = orig_choose_card
    elapsed = time.perf_counter() - start
    print(f'{n_games} donnes simulated in {elapsed:.1f}s -> {len(dataset)} examples '
          f'({len(dataset) / n_games:.1f} per donne)')
    return dataset


def train_supervised(dataset, epochs, lr, batch_size, device='cpu'):
    net = ValueNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(dataset)
    mean_target = sum(r for _, r in dataset) / n
    var_target = sum((r - mean_target) ** 2 for _, r in dataset) / n

    for epoch in range(1, epochs + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch = dataset[start:start + batch_size]
            states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
            targets = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)

            values = net(states)
            loss = F.mse_loss(values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

        mse = total_loss / n
        print(f'epoch {epoch:3d}  mse={mse:8.1f}  r2={1 - mse / var_target:+.4f}')

    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=20000,
                         help="Number of donnes (--policy at seats 0/2 vs "
                              "heuristic) to simulate. Higher than "
                              "pretrain.py (imitation): the counterfactual "
                              "target has a much smaller residual (R2~0.28 "
                              "on the raw return, ~0.03-0.06 after "
                              "subtracting the counterfactual which already "
                              "absorbs most of the signal), so it's slower "
                              "to distinguish from noise -- empirically "
                              "verified data-starved at 5000.")
    parser.add_argument('--epochs', type=int, default=60,
                         help="Number of supervised training epochs. "
                              "Higher than pretrain.py: R2(val) keeps "
                              "improving up to at least 60 on this weak "
                              "residual signal (at 20000 donnes), unlike "
                              "policy imitation which saturates much earlier.")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--policy', default='rl_experiments/imit.pt',
                         help="Policy loaded at seats 0/2 for collection (same role as train.run_episode).")
    parser.add_argument('--out', default='value_imit.pt', help='Path to save the pretrained weights.')
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch is required to pretrain a ValueNet (see requirements.txt).")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    policy = PPOPolicy()
    policy.load(args.policy)

    dataset = collect_dataset(policy, args.games, seed=None)
    net = train_supervised(dataset, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    torch.save(net.state_dict(), args.out)
    print('pretrained weights saved to', args.out)


if __name__ == '__main__':
    main()
