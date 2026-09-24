"""Pretrains a CardNet by supervised imitation of HeuristicPlayer, before
REINFORCE fine-tuning (train.py --load). Simulates heuristic x4 donnes (4
seats, attack and defense combined) to collect (state, chosen card) pairs,
then minimizes cross-entropy to convergence -- deliberately without
early-stopping: the exploration REINFORCE needs is reintroduced afterward via
NeuralPolicy's entropy bonus (cf. remarques_rl.md point 4), not by
under-training here.

Usage:
    python pretrain.py --games 3000 --epochs 20 --out imit.pt
    python train.py --load imit.pt --episodes 5000 --entropy-beta 0.02
"""
import argparse
import random
import time

from coinche.game import GameEngine
from coinche.players import HeuristicPlayer
from coinche.rl_agent import CardNet, CardNetBig, encode_state, _canonical_slots, _slot_index, torch

try:
    import torch.nn.functional as F
except Exception:
    F = None


class RecordingHeuristicPlayer(HeuristicPlayer):
    """Plays exactly like HeuristicPlayer (no duplicated logic); just
    captures (state, chosen card, legal moves) at each decision, into a
    list shared between the 4 seats of the same donne."""
    def __init__(self, name, dataset, ablate_points=False):
        super().__init__(name)
        self.dataset = dataset
        self.ablate_points = ablate_points

    def play_card(self, seat, leader, trick, trump):
        state = encode_state(self, trick, trump, ablate_points=self.ablate_points)
        legal = self.engine.legal_moves(seat, self.hand, trick, trump)
        suits, orders = _canonical_slots(trump)
        legal_idx = [_slot_index(c.suit, c.rank, suits, orders) for c in legal]

        card = super().play_card(seat, leader, trick, trump)

        action_idx = _slot_index(card.suit, card.rank, suits, orders)
        self.dataset.append((state, action_idx, legal_idx))
        return card


def collect_dataset(n_games, seed=None, ablate_points=False):
    if seed is not None:
        random.seed(seed)
    dataset = []
    start = time.perf_counter()
    for i in range(n_games):
        players = [RecordingHeuristicPlayer(f'P{j}', dataset, ablate_points=ablate_points) for j in range(4)]
        engine = GameEngine(players, dealer=i % 4)
        engine.deal()
        engine.run_auction()
        engine.play()
    elapsed = time.perf_counter() - start
    print(f'{n_games} donnes simulated in {elapsed:.1f}s -> {len(dataset)} examples '
          f'({len(dataset) / n_games:.1f} per donne)')
    return dataset


def train_supervised(dataset, epochs, lr, batch_size, device='cpu', net_cls=CardNet):
    net = net_cls().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(dataset)

    for epoch in range(1, epochs + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        total_correct = 0
        for start in range(0, n, batch_size):
            batch = dataset[start:start + batch_size]
            states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
            actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
            mask = torch.full((len(batch), 32), float('-inf'), device=device)
            for i, (_, _, legal_idx) in enumerate(batch):
                mask[i, legal_idx] = 0.0

            logits = net(states) + mask
            loss = F.cross_entropy(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            total_correct += (logits.argmax(dim=1) == actions).sum().item()

        print(f'epoch {epoch:3d}  loss={total_loss / n:.4f}  acc={total_correct / n:.3f}')

    return net


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=3000,
                         help="Number of HeuristicPlayer x4 donnes to simulate to collect examples.")
    parser.add_argument('--epochs', type=int, default=20,
                         help="Number of supervised training epochs (to convergence, see docstring).")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out', default='imit.pt', help='Path to save the pretrained weights.')
    parser.add_argument('--ablate-points-so-far', action='store_true',
                         help="Ablation study: forces encode_state's "
                              "`_points_so_far` block (trick points already "
                              "banked per side) to zero, to pretrain an "
                              "imit.pt that has never seen this feature.")
    parser.add_argument('--architecture', choices=['small', 'big'], default='small',
                         help="'small' = CardNet (1 hidden layer of 128, "
                              "default, already-validated config). 'big' = "
                              "CardNetBig (128/128/64, GELU, remarques_rl.md "
                              "point 6) -- checkpoint incompatible with "
                              "'small', load with train.py --architecture big.")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch is required to pretrain a CardNet (see requirements.txt).")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    net_cls = CardNetBig if args.architecture == 'big' else CardNet
    dataset = collect_dataset(args.games, seed=None, ablate_points=args.ablate_points_so_far)
    net = train_supervised(dataset, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, net_cls=net_cls)

    torch.save(net.state_dict(), args.out)
    print('pretrained weights saved to', args.out)


if __name__ == '__main__':
    main()
