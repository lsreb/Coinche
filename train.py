"""Entraine par REINFORCE une policy de jeu de la carte pour une equipe entiere
(memes poids sur les deux sieges partenaires), face a un adversaire aux sieges
1/3 -- HeuristicPlayer par defaut, une policy figee (self-play), ou un pool
de plusieurs adversaires tire au hasard a chaque episode (voir --opponent).
Les encheres restent gerees par HeuristicPlayer des deux cotes (RLPlayer en
herite) : seul le choix de la carte a jouer, une fois le contrat fixe, est
appris.

Le reward utilise pour la mise a jour REINFORCE n'est pas le reward brut de la
donne (difference de points d'equipe) mais l'ecart avec un contre-factuel :
la meme donne (memes mains, meme donneur, meme contrat garanti puisque les
encheres sont deterministes) rejouee avec l'adversaire de CET episode aux 4
sieges (remarques_rl.md point 3). Ca isole la contribution du jeu de la
policy du hasard de la donne, un signal bien moins bruite qu'une simple
moyenne mobile globale. Voir --no-counterfactual-baseline pour revenir au
reward brut.

Usage:
    python train.py --episodes 5000 --eval-every 200
    python train.py --episodes 2000 --load poids.pt --save poids.pt
    python train.py --load poids.pt --opponent poids.pt --episodes 100000  # self-play vs copie figee
    python train.py --load poids.pt --opponent heuristic,100k.pt,250k.pt --episodes 100000  # pool, tirage uniforme
    python train.py --load poids.pt --opponent heuristic,100k.pt,250k.pt --pfsp --episodes 100000  # pool, tirage PFSP
"""
import argparse
import math
import os
import random
import time

from coinche.game import GameEngine
from coinche.players import RLPlayer, HeuristicPlayer
from coinche.rl_agent import NeuralPolicy, torch

# L'equipe RL controle les sieges 0 et 2 (partenaires) ; 1 et 3 sont l'adversaire.
RL_SEATS = (0, 2)


def _default_opponent(name):
    return HeuristicPlayer(name)


def make_opponent_factory(token):
    """Fabrique de joueur pour un token d'adversaire du pool (--opponent) :
    'heuristic', ou un chemin vers des poids NeuralPolicy sauvegardes
    (self-play contre une copie figee, jeu glouton, jamais mise a jour).
    Au niveau module (pas dans main()) pour etre reutilisable par
    train_ppo.py sans dupliquer cette logique."""
    if token == 'heuristic':
        return _default_opponent
    frozen = NeuralPolicy()
    frozen.load(token)
    frozen.record = False  # glouton, jamais mis a jour (pas d'appel a .update())
    print('adversaire fige charge depuis', token)
    return lambda name, frozen=frozen: RLPlayer(name, frozen)


def build_opponent_pool(opponent_arg):
    """Parse --opponent ('token1,token2,...') en une liste [(token, factory)].
    Leve une erreur si vide."""
    tokens = [t.strip() for t in opponent_arg.split(',') if t.strip()]
    if not tokens:
        raise SystemExit("--opponent doit contenir au moins une valeur.")
    return [(t, make_opponent_factory(t)) for t in tokens]


class PFSPSampler:
    """Tirage de l'adversaire de type Prioritized Fictitious Self-Play
    (AlphaStar) : plutot qu'un tirage uniforme dans le pool (--opponent), les
    adversaires les plus coriaces pour la policy en cours (win-rate courant
    le plus bas) sont favorises, via une repartition de Boltzmann sur
    `1 - win_rate` (temperature `temperature` : plus bas = biais plus marque
    vers le plus coriace, plus haut = plus proche d'un tirage uniforme). Le
    win-rate par adversaire est une moyenne mobile exponentielle
    (`ema_beta`, initialisee a 0.5 = pas d'info) mise a jour a chaque episode
    reellement joue contre lui (`record_outcome`). La repartition de tirage
    n'est recalculee que tous les `refresh_every` episodes (pas a chaque
    episode) pour rester lisible et stable ; entre deux recalculs le tirage
    utilise les poids figes du dernier recalcul. `picks` compte combien de
    fois chaque adversaire a ete choisi, pour reporting (cf. `summary`)."""

    def __init__(self, pool, refresh_every=5000, temperature=1.0, ema_beta=0.98):
        if not pool:
            raise ValueError('pool vide')
        self.names = [name for name, _ in pool]
        self.factories = [factory for _, factory in pool]
        self.refresh_every = refresh_every
        self.temperature = max(temperature, 1e-6)
        self.ema_beta = ema_beta
        self.ema_winrate = {name: 0.5 for name in self.names}
        self.picks = {name: 0 for name in self.names}
        self._weights = [1.0 / len(self.names)] * len(self.names)

    def _refresh_weights(self):
        toughness = [1.0 - self.ema_winrate[name] for name in self.names]
        m = max(toughness)  # stabilite numerique du softmax (invariance par decalage)
        exps = [math.exp((t - m) / self.temperature) for t in toughness]
        s = sum(exps)
        self._weights = [e / s for e in exps]

    def choose(self, global_ep):
        if len(self.names) > 1 and (global_ep == 1 or global_ep % self.refresh_every == 0):
            self._refresh_weights()
        if len(self.names) == 1:
            idx = 0
        else:
            idx = random.choices(range(len(self.names)), weights=self._weights, k=1)[0]
        name = self.names[idx]
        self.picks[name] += 1
        return name, self.factories[idx]

    def record_outcome(self, name, reward):
        won = 1.0 if reward > 0 else 0.0
        self.ema_winrate[name] = self.ema_beta * self.ema_winrate[name] + (1 - self.ema_beta) * won

    def summary(self):
        return '  '.join(
            f"{name}:{self.picks[name]}x(wr={self.ema_winrate[name]:.2f})" for name in self.names
        )


def run_episode(policy, dealer, opponent_factory=_default_opponent):
    """Joue une donne avec la policy RL aux sieges 0/2, contre `opponent_factory`
    aux sieges 1/3. Retourne le reward brut (difference de points d'equipe,
    cf. `evaluate`) et les mains distribuees (pour rejouer eventuellement la
    meme donne en contre-factuel avec ce meme adversaire)."""
    players = [
        RLPlayer('P0', policy), opponent_factory('P1'),
        RLPlayer('P2', policy), opponent_factory('P3'),
    ]
    engine = GameEngine(players, dealer=dealer)
    engine.deal()
    engine.run_auction()
    team_points, _contract = engine.play()
    reward = team_points[0] - team_points[1]  # equipe RL = siege 0/2, parite 0
    return reward, engine.history['deal_hands']


def counterfactual_reward(hands, dealer, opponent_factory=_default_opponent):
    """Rejoue exactement la meme donne (memes mains, meme donneur) avec
    `opponent_factory` aux 4 sieges : reference bas-bruit pour isoler la
    contribution du jeu de la policy par rapport a ce que cet adversaire
    aurait fait sur les memes cartes (remarques_rl.md point 3), plutot que de
    se comparer a une moyenne globale que le hasard de la donne rend bruitee.
    Doit toujours recevoir le meme `opponent_factory` que le `run_episode` qui
    a produit `hands` (cf. la boucle d'entrainement dans `main`) : avec un
    pool d'adversaires, comparer a un adversaire different de celui reellement
    affronte cet episode melangerait deux signaux incoherents. bid()
    (Heuristic comme RLPlayer, qui l'herite tel quel) n'a aucun alea : le
    contrat reproduit est garanti identique a celui de la donne d'origine."""
    players = [opponent_factory(f'H{i}') for i in range(4)]
    engine = GameEngine(players, dealer=dealer)
    engine.deal(hands=hands)
    engine.run_auction()
    team_points, _contract = engine.play()
    return team_points[0] - team_points[1]


def evaluate(policy, n_games, start_dealer=0, opponent_factory=_default_opponent):
    """Evalue en mode glouton contre `opponent_factory` (HeuristicPlayer par
    defaut, y compris pendant un entrainement en pool -- cf. `main` -- pour
    garder un suivi de progression comparable d'un run a l'autre) : reward
    moyenne et taux de victoire (fraction des donnes ou l'equipe RL marque
    plus que l'adversaire)."""
    was_recording = policy.record
    policy.record = False
    total = 0.0
    wins = 0
    for i in range(n_games):
        reward, _hands = run_episode(policy, dealer=(start_dealer + i) % 4, opponent_factory=opponent_factory)
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
    parser.add_argument('--no-counterfactual-baseline', action='store_true',
                         help="Desactive le contre-factuel (remarques_rl.md point 3) : revient a "
                              "utiliser le reward brut directement, avec la seule baseline EMA de NeuralPolicy.")
    parser.add_argument('--opponent', default='heuristic',
                         help="Adversaire(s) aux sieges 1/3 (et pour le contre-factuel), separes par des virgules : "
                              "'heuristic' et/ou des chemins vers des poids NeuralPolicy sauvegardes (self-play "
                              "contre une copie figee, jeu glouton, jamais mise a jour). Avec plusieurs valeurs, "
                              "un adversaire est tire au hasard a chaque episode (pool plutot qu'un adversaire "
                              "fixe unique) ; l'eval loggee pendant l'entrainement reste toujours vs heuristic "
                              "pour un suivi comparable d'un run a l'autre.")
    parser.add_argument('--entropy-decay', choices=['none', 'invsqrt', 'inv'], default='none',
                         help="Decroissance du bonus d'entropie au fil des episodes : 'none' (constant), "
                              "'invsqrt' (beta/sqrt(ep)), 'inv' (beta/ep).")
    parser.add_argument('--pfsp', action='store_true',
                         help="Prioritized Fictitious Self-Play : biaise le tirage de --opponent vers les "
                              "adversaires les plus coriaces (win-rate courant le plus bas) au lieu d'un "
                              "tirage uniforme (cf. PFSPSampler). Sans effet si --opponent n'a qu'une valeur.")
    parser.add_argument('--pfsp-refresh-every', type=int, default=5000,
                         help="Frequence (en episodes) de recalcul de la repartition de tirage PFSP.")
    parser.add_argument('--pfsp-temperature', type=float, default=1.0,
                         help="Temperature du softmax PFSP : plus bas = biais plus marque vers l'adversaire "
                              "le plus coriace, plus haut = plus proche d'un tirage uniforme.")
    parser.add_argument('--pfsp-ema-beta', type=float, default=0.98,
                         help="Coefficient de la moyenne mobile exponentielle du win-rate par adversaire "
                              "(PFSP) : plus proche de 1 = memoire plus longue.")
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

    opponent_pool = build_opponent_pool(args.opponent)
    sampler = PFSPSampler(opponent_pool, refresh_every=args.pfsp_refresh_every,
                           temperature=args.pfsp_temperature, ema_beta=args.pfsp_ema_beta) if args.pfsp else None

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
        if sampler is not None:
            opponent_name, opponent_factory = sampler.choose(global_ep)
        else:
            opponent_name, opponent_factory = (
                opponent_pool[0] if len(opponent_pool) == 1 else random.choice(opponent_pool)
            )
        reward, hands = run_episode(policy, dealer=dealer, opponent_factory=opponent_factory)
        if sampler is not None:
            sampler.record_outcome(opponent_name, reward)
        if args.no_counterfactual_baseline:
            training_reward = reward
        else:
            training_reward = reward - counterfactual_reward(hands, dealer=dealer, opponent_factory=opponent_factory)
        policy.update(training_reward)

        window.append(reward)  # reward brut, pour un suivi interpretable (vs l'adversaire)
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
            if sampler is not None:
                print(f"  picks: {sampler.summary()}")

        if args.checkpoint_every and args.checkpoint_dir and ep % args.checkpoint_every == 0:
            policy.save(os.path.join(args.checkpoint_dir, f'ckpt_ep{global_ep}.pt'))

    if args.save:
        policy.save(args.save)
        print('poids sauvegardes dans', args.save)

    if sampler is not None:
        print('PFSP picks total:', sampler.summary())


if __name__ == '__main__':
    main()
