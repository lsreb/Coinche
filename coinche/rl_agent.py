import random
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None

class SimplePolicy:
    """A minimal tabular/ML-agnostic policy placeholder.

    The `choose_card` method must return an index in the hand list.
    """
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)

    def choose_card(self, hand, leader, trick, trump):
        return self.rng.randrange(len(hand))

# A simple neural policy sketch (optional, requires torch)
if torch is not None:
    class CardNet(nn.Module):
        def __init__(self, in_dim=16, hidden=64):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    class NeuralPolicy:
        def __init__(self, device='cpu'):
            self.net = CardNet().to(device)
            self.device = device

        def card_features(self, card):
            # simple encoding: suit one-hot (4) + rank one-hot (8) = 12 dims, pad to 16
            suit_idx = 'PCKT'.index(card.suit) if card.suit in 'PCKT' else 0
            rank_idx = ['A','10','K','Q','J','9','8','7'].index(card.rank)
            v = [0]*16
            v[suit_idx] = 1
            v[4+rank_idx] = 1
            return v

        def choose_card(self, hand, leader, trick, trump):
            feats = [self.card_features(c) for c in hand]
            import torch
            x = torch.tensor(feats, dtype=torch.float32, device=self.device)
            scores = self.net(x).squeeze(-1)
            probs = torch.softmax(scores, dim=0).cpu().detach().numpy()
            idx = int(torch.multinomial(torch.tensor(probs),1).item())
            return idx
