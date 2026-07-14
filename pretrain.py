"""Pre-entraine un CardNet par imitation supervisee de HeuristicPlayer, avant le
fine-tuning REINFORCE (train.py --load). Simule des donnes heuristique x4 (4
sieges, attaque et defense confondues) pour collecter des paires (etat, carte
choisie), puis minimise l'entropie croisee jusqu'a convergence -- volontairement
sans early-stopping : l'exploration necessaire au REINFORCE est reintroduite
ensuite via le bonus d'entropie de NeuralPolicy (cf. remarques_rl.md point 4),
pas en sous-entrainant ici.

Usage:
    python pretrain.py --games 3000 --epochs 20 --out imit.pt
    python train.py --load imit.pt --episodes 5000 --entropy-beta 0.02
"""
import argparse
import random
import time

from coinche.game import GameEngine
from coinche.players import HeuristicPlayer
from coinche.rl_agent import CardNet, encode_state, _canonical_slots, _slot_index, torch

try:
    import torch.nn.functional as F
except Exception:
    F = None


class RecordingHeuristicPlayer(HeuristicPlayer):
    """Joue exactement comme HeuristicPlayer (aucune logique dupliquee) ; capture
    juste (etat, carte choisie, coups legaux) a chaque decision, dans une liste
    partagee entre les 4 sieges d'une meme donne."""
    def __init__(self, name, dataset):
        super().__init__(name)
        self.dataset = dataset

    def play_card(self, seat, leader, trick, trump):
        state = encode_state(self, trick, trump)
        legal = self.engine.legal_moves(seat, self.hand, trick, trump)
        suits, orders = _canonical_slots(trump)
        legal_idx = [_slot_index(c.suit, c.rank, suits, orders) for c in legal]

        card = super().play_card(seat, leader, trick, trump)

        action_idx = _slot_index(card.suit, card.rank, suits, orders)
        self.dataset.append((state, action_idx, legal_idx))
        return card


def collect_dataset(n_games, seed=None):
    if seed is not None:
        random.seed(seed)
    dataset = []
    start = time.perf_counter()
    for i in range(n_games):
        players = [RecordingHeuristicPlayer(f'P{j}', dataset) for j in range(4)]
        engine = GameEngine(players, dealer=i % 4)
        engine.deal()
        engine.run_auction()
        engine.play()
    elapsed = time.perf_counter() - start
    print(f'{n_games} donnes simulees en {elapsed:.1f}s -> {len(dataset)} exemples '
          f'({len(dataset) / n_games:.1f} par donne)')
    return dataset


def train_supervised(dataset, epochs, lr, batch_size, device='cpu'):
    net = CardNet().to(device)
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
                         help="Nombre de donnes HeuristicPlayer x4 a simuler pour collecter les exemples.")
    parser.add_argument('--epochs', type=int, default=20,
                         help="Nombre d'epochs d'entrainement supervise (jusqu'a convergence, voir docstring).")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out', default='imit.pt', help='Chemin de sauvegarde des poids pre-entraines.')
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch est requis pour pre-entrainer un CardNet (voir requirements.txt).")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    dataset = collect_dataset(args.games, seed=None)
    net = train_supervised(dataset, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    torch.save(net.state_dict(), args.out)
    print('poids pre-entraines sauvegardes dans', args.out)


if __name__ == '__main__':
    main()
