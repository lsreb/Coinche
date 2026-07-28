"""Evalue un ou plusieurs checkpoints en mode glouton sur N parties contre
HeuristicPlayer (memes sieges/equipe que train.py : policy aux sieges 0/2).
Charge via PPOPolicy (train_ppo.py), dont `load()` accepte aussi bien un
checkpoint CardNet (REINFORCE, train.py) qu'un checkpoint ActorCriticNet
(PPO) -- en greedy (`record=False`) seule la tete policy compte, la tete
valeur (absente/aleatoire sur un checkpoint REINFORCE) n'intervient pas.

Contrairement a une version precedente qui reseedait `random` avant chaque
checkpoint et laissait chaque partie se dealer fraichement (`engine.deal()`
sans mains fournies), CETTE version genere les N (mains, dealer) une seule
fois -- INDEPENDAMMENT de tout checkpoint -- puis rejoue exactement ces memes
mains pour chacun via `engine.deal(hands=...)`. La version precedente ne
garantissait PAS des mains identiques au-dela des tout premiers essais : des
qu'un checkpoint joue une carte differente d'un autre a un moment de la
partie, HeuristicPlayer._choose_defausse_suit() (le seul point d'alea de
HeuristicPlayer pendant le jeu, hors encheres) peut consommer un nombre
different de tirages `random`, ce qui fait diverger tout le `random` global
pour le reste de la boucle -- verifie empiriquement : divergence des mains
des la 4e partie sur 100 entre deux checkpoints avec le meme seed de depart.

Second piege corrige : si les mains generees menent a un "tout le monde
passe", `GameEngine.run_auction()` redonne EN INTERNE avec `self.deal()` SANS
argument (game.py:139-143) -- des mains ALEATOIRES qui ecrasent celles
fournies, silencieusement. `generate_fixed_deals()` filtre ces mains-la a la
generation (verifie via un auction jetable Heuristic x4, bid() etant
alea-free et identique a ce que fait RLPlayer qui l'herite tel quel) pour que
le rejouage a mains fixes ne puisse jamais retomber dans ce cas.

Le pseudo-checkpoint special "heuristic" evalue HeuristicPlayer contre
HeuristicPlayer aux 4 sieges (aucune policy chargee) : sert de reference pour
savoir quel win_rate/ecart de points resulte du seul hasard de la donne
(dealer, mains) quand les deux camps jouent strictement la meme strategie.

Usage:
    python eval_policy.py --games 1500 heuristic rl_experiments/imit.pt rl_experiments/exp_lowlr_lowent/final.pt
"""
import argparse
import random

from coinche.game import GameEngine
from coinche.players import HeuristicPlayer, RLPlayer
from coinche.rl_agent import CardNet, CardNetBig
from train_ppo import PPOPolicy, SharedTrunkPPOPolicy, SharedTrunkAuxPPOPolicy, torch


def _leads_to_all_pass(hands, dealer):
    """True si ces mains, auctionnees par 4 HeuristicPlayer (bid() alea-free,
    identique pour RLPlayer qui l'herite tel quel -- donc ce test vaut pour
    n'importe quel checkpoint aux sieges 0/2), font passer tout le monde et
    declenchent la redonne interne de game.py:139-143."""
    players = [HeuristicPlayer(f'D{i}') for i in range(4)]
    engine = GameEngine(players, dealer=dealer)
    engine.deal(hands=hands)
    engine.run_auction()
    return engine.history['deal_hands'] != hands


def generate_fixed_deals(n_games, seed):
    """Genere n_games (mains, dealer) une seule fois, indep. de tout
    checkpoint, en ecartant les mains qui provoqueraient une redonne interne
    (cf. docstring du module) -- pour que `deal(hands=...)` donne exactement
    les memes cartes a chaque checkpoint evalue avec cette liste."""
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
    random.seed(seed)  # meme point de depart pour l'alea residuel de Heuristic (defausse)
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
    parser.add_argument('checkpoints', nargs='+', help='Chemins des poids a evaluer (ou "heuristic").')
    parser.add_argument('--games', type=int, default=1500, help='Nombre de donnes (partagees par tous les checkpoints).')
    parser.add_argument('--seed', type=int, default=42,
                         help='Seed pour generer les donnes fixes (une fois) et reappliquee avant chaque checkpoint.')
    parser.add_argument('--ablate-points-so-far', action='store_true',
                         help="Applique le meme flag d'ablation qu'a l'entrainement (train_ppo.py/pretrain.py "
                              "--ablate-points-so-far) a TOUS les checkpoints de cet appel -- pour comparer un "
                              "groupe ablate a un autre, lancer eval_policy.py une fois par groupe (memes "
                              "--games/--seed => memes donnes generees, cf. generate_fixed_deals).")
    parser.add_argument('--architecture', choices=['small', 'big', 'shared', 'shared_aux'], default='small',
                         help="'small' = CardNet (defaut), 'big' = CardNetBig, 'shared' = "
                              "SharedTrunkActorCritic, 'shared_aux' = SharedTrunkActorCriticAux "
                              "(remarques_rl.md point 6, rl_experiments_v5) -- s'applique a TOUS les "
                              "checkpoints non-heuristic de cet appel ; melanger plusieurs architectures "
                              "necessite un appel separe par groupe (meme --games/--seed => memes donnes).")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch est requis pour evaluer une NeuralPolicy (voir requirements.txt).")

    deals = generate_fixed_deals(args.games, args.seed)
    policy_net_cls = CardNetBig if args.architecture == 'big' else CardNet

    results = []
    for path in args.checkpoints:
        if path == 'heuristic':
            factory = lambda: [HeuristicPlayer(f'H{i}') for i in range(4)]
        else:
            if args.architecture == 'shared_aux':
                policy = SharedTrunkAuxPPOPolicy()
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
    print("Resume (memes %d donnes, fixees une fois, pour tous les checkpoints, seed=%d) :"
          % (args.games, args.seed))
    for path, avg, win_rate in sorted(results, key=lambda r: -r[1]):
        print(f"  {path:55s}  eval_avg={avg:+7.2f}  win_rate={100*win_rate:5.1f}%")


if __name__ == '__main__':
    main()
