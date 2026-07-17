"""Entraine par REINFORCE une policy de jeu de la carte pour une equipe entiere
(memes poids sur les deux sieges partenaires), face a une equipe HeuristicPlayer.
Les encheres restent geree par HeuristicPlayer des deux cotes (RLPlayer en herite) :
seul le choix de la carte a jouer, une fois le contrat fixe, est appris.

Le reward utilise pour la mise a jour REINFORCE n'est pas le reward brut de la
donne (difference de points d'equipe) mais l'ecart avec un contre-factuel :
la meme donne (memes mains, meme donneur, meme contrat garanti puisque les
encheres sont deterministes) rejouee avec HeuristicPlayer aux 4 sieges
(remarques_rl.md point 3). Ca isole la contribution du jeu de la policy du
hasard de la donne, un signal bien moins bruite qu'une simple moyenne mobile
globale. Voir --no-heuristic-baseline pour revenir au reward brut.

Usage:
    python train.py --episodes 5000 --eval-every 200
    python train.py --episodes 2000 --load poids.pt --save poids.pt
"""
import argparse
import os
import random
import time

from coinche.game import GameEngine
from coinche.players import RLPlayer, HeuristicPlayer
from coinche.rl_agent import NeuralPolicy, torch

# L'equipe RL controle les sieges 0 et 2 (partenaires) ; 1 et 3 restent heuristiques.
RL_SEATS = (0, 2)


def run_episode(policy, dealer):
    """Joue une donne avec la policy RL aux sieges 0/2. Retourne le reward brut
    (difference de points d'equipe, cf. `evaluate`) et les mains distribuees
    (pour rejouer eventuellement la meme donne en contre-factuel heuristique)."""
    players = [
        RLPlayer('P0', policy), HeuristicPlayer('P1'),
        RLPlayer('P2', policy), HeuristicPlayer('P3'),
    ]
    engine = GameEngine(players, dealer=dealer)
    engine.deal()
    engine.run_auction()
    team_points, _contract = engine.play()
    reward = team_points[0] - team_points[1]  # equipe RL = siege 0/2, parite 0
    return reward, engine.history['deal_hands']


def heuristic_counterfactual_reward(hands, dealer):
    """Rejoue exactement la meme donne (memes mains, meme donneur) avec
    HeuristicPlayer aux 4 sieges : reference bas-bruit pour isoler la
    contribution du jeu de la policy par rapport a un jeu heuristique de
    reference sur les memes cartes (remarques_rl.md point 3), plutot que de se
    comparer a une moyenne globale que le hasard de la donne rend bruitee.
    HeuristicPlayer.bid() n'a aucun alea : le contrat reproduit est garanti
    identique a celui de la donne RL d'origine."""
    players = [HeuristicPlayer(f'H{i}') for i in range(4)]
    engine = GameEngine(players, dealer=dealer)
    engine.deal(hands=hands)
    engine.run_auction()
    team_points, _contract = engine.play()
    return team_points[0] - team_points[1]


def evaluate(policy, n_games, start_dealer=0):
    """Evalue en mode glouton : reward moyenne (vs heuristique) et taux de
    victoire (fraction des donnes ou l'equipe RL marque plus que l'adversaire)."""
    was_recording = policy.record
    policy.record = False
    total = 0.0
    wins = 0
    for i in range(n_games):
        reward, _hands = run_episode(policy, dealer=(start_dealer + i) % 4)
        total += reward
        if reward > 0:
            wins += 1
    policy.record = was_recording
    avg = total / n_games if n_games else 0.0
    win_rate = wins / n_games if n_games else 0.0
    return avg, win_rate


def entropy_beta_for_episode(base_beta, ep, decay):
    """Coefficient d'entropie effectif a l'episode `ep` (>=1) selon le schema de
    decroissance choisi : 'none' (constant), 'invsqrt' (base/sqrt(ep)), ou
    'inv' (base/ep)."""
    if decay == 'none':
        return base_beta
    if decay == 'invsqrt':
        return base_beta / (ep ** 0.5)
    if decay == 'inv':
        return base_beta / ep
    raise ValueError(f'schema de decroissance inconnu: {decay}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=2000, help='Nombre de donnes d\'entrainement.')
    parser.add_argument('--eval-every', type=int, default=200, help='Frequence (en episodes) des evaluations gloutonnes.')
    parser.add_argument('--eval-games', type=int, default=100, help='Nombre de donnes par evaluation.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--entropy-beta', type=float, default=0.01,
                         help="Poids du bonus d'entropie dans la loss REINFORCE (force l'exploration, "
                              "utile en particulier apres un --load d'une policy pre-entrainee par imitation).")
    parser.add_argument('--no-heuristic-baseline', action='store_true',
                         help="Desactive le contre-factuel heuristique (remarques_rl.md point 3) : revient a "
                              "utiliser le reward brut directement, avec la seule baseline EMA de NeuralPolicy.")
    parser.add_argument('--entropy-decay', choices=['none', 'invsqrt', 'inv'], default='none',
                         help="Decroissance du bonus d'entropie au fil des episodes : 'none' (constant), "
                              "'invsqrt' (beta/sqrt(ep)), 'inv' (beta/ep).")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save', default=None, help="Chemin pour sauvegarder les poids en fin d'entrainement.")
    parser.add_argument('--load', default=None, help='Chemin pour reprendre depuis des poids sauvegardes.')
    parser.add_argument('--checkpoint-every', type=int, default=None,
                         help="Sauvegarde un checkpoint tous les N episodes, en plus de --save en fin d'entrainement.")
    parser.add_argument('--checkpoint-dir', default=None,
                         help='Dossier de sauvegarde des checkpoints (requis avec --checkpoint-every).')
    parser.add_argument('--episode-offset', type=int, default=0,
                         help="Decalage ajoute au compteur d'episode (logs, dealer, decroissance d'entropie, "
                              "nom des checkpoints) : pour reprendre un entrainement precedent avec une "
                              "numerotation absolue coherente, mettre le nombre d'episodes deja effectues.")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch est requis pour entrainer une NeuralPolicy (voir requirements.txt).")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    policy = NeuralPolicy(lr=args.lr, entropy_beta=args.entropy_beta)
    if args.load:
        policy.load(args.load)
        print('poids charges depuis', args.load)

    if args.checkpoint_every and args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    window = []
    adv_window = []
    start = time.perf_counter()
    for ep in range(1, args.episodes + 1):
        global_ep = ep + args.episode_offset
        policy.entropy_beta = entropy_beta_for_episode(args.entropy_beta, global_ep, args.entropy_decay)
        dealer = global_ep % 4
        reward, hands = run_episode(policy, dealer=dealer)
        if args.no_heuristic_baseline:
            training_reward = reward
        else:
            training_reward = reward - heuristic_counterfactual_reward(hands, dealer=dealer)
        policy.update(training_reward)

        window.append(reward)  # reward brut, pour un suivi interpretable (vs heuristique)
        adv_window.append(training_reward)  # signal reellement utilise pour la mise a jour
        if len(window) > 200:
            window.pop(0)
            adv_window.pop(0)

        if ep % args.eval_every == 0:
            avg_train = sum(window) / len(window)
            avg_adv = sum(adv_window) / len(adv_window)
            avg_eval, win_rate = evaluate(policy, args.eval_games, start_dealer=global_ep)
            elapsed = time.perf_counter() - start
            print(f"episode {global_ep:6d}  train_avg={avg_train:+7.1f}  adv_avg={avg_adv:+7.1f}  "
                  f"eval_avg({args.eval_games})={avg_eval:+7.1f}  win_rate={100*win_rate:5.1f}%  "
                  f"entropy_beta={policy.entropy_beta:.4f}  baseline={policy.baseline:+7.1f}  ({elapsed:.1f}s)")

        if args.checkpoint_every and args.checkpoint_dir and ep % args.checkpoint_every == 0:
            policy.save(os.path.join(args.checkpoint_dir, f'ckpt_ep{global_ep}.pt'))

    if args.save:
        policy.save(args.save)
        print('poids sauvegardes dans', args.save)


if __name__ == '__main__':
    main()
