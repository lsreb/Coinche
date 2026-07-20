"""Pre-entraine un ValueNet (train_ppo.py) par regression supervisee du retour
CONTRE-FACTUEL -- la meme quantite que train_ppo.py entraine reellement (cf.
`train.counterfactual_reward`, remarques_rl.md point 3), PAS le retour brut --
avant le fine-tuning PPO (train_ppo.py --load-value).

Une premiere version de ce script regressait sur le retour BRUT
(team_points[0]-team_points[1]), collecte sur des donnes HeuristicPlayer x4 :
bon R2 hors ligne (0.28), mais performance finale APRES fine-tuning PPO moins
bonne qu'un ValueNet initialise aleatoirement (verifie empiriquement). Cause :
train_ppo.py n'entraine value_loss (et l'avantage) que sur
`reward - counterfactual_reward(...)`, l'ecart a ce qu'un heuristique aurait
fait sur les MEMES cartes -- et ce contre-factuel absorbe presque tout le
signal `is_attacker` du retour brut (verifie : ecart attaque/defense d'environ
190 points sur le retour brut, ~17 points seulement sur le retour
contre-factuel, sur les memes donnes). Le ValueNet pre-entraine sur le retour
brut demarrait donc avec des predictions confiantes mais systematiquement
biaisees pour la VRAIE cible PPO (avantage artificiellement trop negatif en
attaque, trop positif en defense) -- pire qu'un depart neutre.

Simule donc des donnes avec `--policy` (imit.pt par defaut, meme si pas
encore fine-tunee par PPO) aux sieges 0/2 contre HeuristicPlayer aux sieges
1/3 -- exactement comme `train.run_episode` -- calcule le meme contre-factuel
que train_ppo.py sur les memes mains, et regresse ValueNet sur cette cible.

Usage:
    python pretrain_value.py --games 5000 --epochs 20 --out value_imit.pt
    python train_ppo.py --load rl_experiments/imit.pt --load-value value_imit.pt --episodes 50000
"""
import argparse
import random
import time

from coinche.game import GameEngine
from coinche.players import HeuristicPlayer, RLPlayer
from coinche.rl_agent import encode_state, torch
from train import counterfactual_reward
from train_ppo import PPOPolicy, ValueNet

try:
    import torch.nn.functional as F
except Exception:
    F = None


def collect_dataset(policy, n_games, seed=None):
    """Simule n_games donnes, `policy` aux sieges 0/2 (comme train.run_episode)
    contre HeuristicPlayer aux sieges 1/3, et associe a chaque etat rencontre
    aux sieges 0/2 le retour contre-factuel de la donne entiere (meme
    convention que train_ppo.py : un seul scalaire pour tous les etats de la
    donne, recompense terminale, cf. docstring du module)."""
    if seed is not None:
        random.seed(seed)
    dataset = []
    current = []

    orig_choose_card = policy.choose_card
    def instrumented(player, legal, leader, trick, trump):
        current.append(encode_state(player, trick, trump))
        return orig_choose_card(player, legal, leader, trick, trump)
    policy.choose_card = instrumented
    policy.record = False  # etats representatifs de la meilleure reponse de la policy, pas d'exploration bruitee

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
    print(f'{n_games} donnes simulees en {elapsed:.1f}s -> {len(dataset)} exemples '
          f'({len(dataset) / n_games:.1f} par donne)')
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
                         help="Nombre de donnes (--policy aux sieges 0/2 vs heuristic) a simuler. Plus eleve "
                              "que pretrain.py (imitation) : la cible contre-factuelle a un residuel bien "
                              "plus faible (R2~0.28 sur le retour brut, ~0.03-0.06 apres soustraction du "
                              "contrefactuel qui absorbe deja la plus grosse partie du signal), donc plus "
                              "lent a distinguer du bruit -- verifie empiriquement data-starved a 5000.")
    parser.add_argument('--epochs', type=int, default=60,
                         help="Nombre d'epochs d'entrainement supervise. Plus eleve que pretrain.py : R2(val) "
                              "continue de progresser jusqu'a au moins 60 sur ce residuel faible signal (a "
                              "20000 donnes), contrairement a l'imitation de policy qui sature bien plus tot.")
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--policy', default='rl_experiments/imit.pt',
                         help="Policy chargee aux sieges 0/2 pour la collecte (meme role que train.run_episode).")
    parser.add_argument('--out', default='value_imit.pt', help='Chemin de sauvegarde des poids pre-entraines.')
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch est requis pour pre-entrainer un ValueNet (voir requirements.txt).")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    policy = PPOPolicy()
    policy.load(args.policy)

    dataset = collect_dataset(policy, args.games, seed=None)
    net = train_supervised(dataset, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    torch.save(net.state_dict(), args.out)
    print('poids pre-entraines sauvegardes dans', args.out)


if __name__ == '__main__':
    main()
