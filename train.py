"""Trains a card-play policy via REINFORCE for a whole team (same weights on
both partner seats), against an opponent at seats 1/3 -- HeuristicPlayer by
default, a frozen policy (self-play), or a pool of several opponents sampled
at random each episode (see --opponent). Bidding stays handled by
HeuristicPlayer on both sides (RLPlayer inherits it as-is): only the choice
of which card to play, once the contract is fixed, is learned.

The reward used for the REINFORCE update isn't the donne's raw reward (team
point difference) but the gap to a counterfactual: the same donne (same
hands, same dealer, same guaranteed contract since bidding is deterministic)
replayed with THIS episode's opponent at all 4 seats (remarques_rl.md point
3). This isolates the policy's own contribution to play from the donne's
luck, a signal much less noisy than a simple global moving average. See
--no-counterfactual-baseline to fall back to the raw reward.

Usage:
    python train.py --episodes 5000 --eval-every 200
    python train.py --episodes 2000 --load weights.pt --save weights.pt
    python train.py --load weights.pt --opponent weights.pt --episodes 100000  # self-play vs frozen copy
    python train.py --load weights.pt --opponent heuristic,100k.pt,250k.pt --episodes 100000  # pool, uniform sampling
    python train.py --load weights.pt --opponent heuristic,100k.pt,250k.pt --pfsp --episodes 100000  # pool, PFSP sampling
"""
import argparse
import math
import os
import random
import time

from coinche.game import GameEngine
from coinche.players import RLPlayer, HeuristicPlayer
from coinche.rl_agent import NeuralPolicy, CardNet, CardNetBig, torch

# The RL team controls seats 0 and 2 (partners); 1 and 3 are the opponent.
RL_SEATS = (0, 2)


def _default_opponent(name):
    return HeuristicPlayer(name)


def make_opponent_factory(token, net_cls=CardNet):
    """Player factory for a pool opponent token (--opponent):
    'heuristic', or a path to saved NeuralPolicy weights (self-play
    against a frozen copy, greedy play, never updated). `net_cls` must
    match the loaded checkpoint's architecture (e.g. CardNetBig for a
    growing pool in --architecture big, cf. run_growing_pool.py) --
    otherwise load_state_dict fails (mismatched keys). Kept at module
    level (not inside main()) so it's reusable by train_ppo.py without
    duplicating this logic."""
    if token == 'heuristic':
        return _default_opponent
    frozen = NeuralPolicy(net_cls=net_cls)
    frozen.load(token)
    frozen.record = False  # greedy, never updated (no call to .update())
    print('frozen opponent loaded from', token)
    return lambda name, frozen=frozen: RLPlayer(name, frozen)


def build_opponent_pool(opponent_arg, net_cls=CardNet):
    """Parses --opponent ('token1,token2,...') into a list [(token, factory)].
    Raises an error if empty."""
    tokens = [t.strip() for t in opponent_arg.split(',') if t.strip()]
    if not tokens:
        raise SystemExit("--opponent must contain at least one value.")
    return [(t, make_opponent_factory(t, net_cls=net_cls)) for t in tokens]


class PFSPSampler:
    """Prioritized Fictitious Self-Play (AlphaStar-style) opponent
    sampling: rather than sampling uniformly from the pool (--opponent),
    the toughest opponents for the current policy (lowest current win
    rate) are favored, via a Boltzmann distribution over `1 - win_rate`
    (temperature `temperature`: lower = stronger bias toward the
    toughest, higher = closer to uniform sampling). The per-opponent win
    rate is an exponential moving average (`ema_beta`, initialized at 0.5
    = no info) updated on every episode actually played against them
    (`record_outcome`). The sampling distribution is only recomputed
    every `refresh_every` episodes (not every episode) to stay readable
    and stable; between two recomputes, sampling uses the frozen weights
    from the last recompute. `picks` counts how many times each opponent
    was chosen, for reporting (cf. `summary`).

    `explore_eps` (2026-07-30 discussion): exploration floor -- mixes
    the softmax with uniform sampling (`final_weight =
    (1-explore_eps)*softmax + explore_eps/N`), to guarantee that no
    opponent falls below `explore_eps/N` probability even at a very low
    temperature. Without this, an aggressive temperature (e.g. 0.05) can
    leave an easy opponent at under 1% of games (computed by hand that
    day: a toughness gap of 0.20 with temperature=0.05 gives
    exp(-4)=1.8%, i.e. ~0.2% once normalized over 10 opponents) -- to the
    point of never training against it again. Default 0.0 = no floor,
    unchanged behavior."""

    def __init__(self, pool, refresh_every=5000, temperature=1.0, ema_beta=0.98, explore_eps=0.0):
        if not pool:
            raise ValueError('empty pool')
        self.names = [name for name, _ in pool]
        self.factories = [factory for _, factory in pool]
        self.refresh_every = refresh_every
        self.temperature = max(temperature, 1e-6)
        self.ema_beta = ema_beta
        self.explore_eps = explore_eps
        self.ema_winrate = {name: 0.5 for name in self.names}
        self.picks = {name: 0 for name in self.names}
        self._weights = [1.0 / len(self.names)] * len(self.names)

    def _refresh_weights(self):
        toughness = [1.0 - self.ema_winrate[name] for name in self.names]
        m = max(toughness)  # softmax numerical stability (shift invariance)
        exps = [math.exp((t - m) / self.temperature) for t in toughness]
        s = sum(exps)
        n = len(self.names)
        self._weights = [(1 - self.explore_eps) * (e / s) + self.explore_eps / n for e in exps]

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
    """Plays one donne with the RL policy at seats 0/2, against
    `opponent_factory` at seats 1/3. Returns the raw reward (team point
    difference, cf. `evaluate`) and the dealt hands (to possibly replay
    the same donne as a counterfactual with this same opponent)."""
    players = [
        RLPlayer('P0', policy), opponent_factory('P1'),
        RLPlayer('P2', policy), opponent_factory('P3'),
    ]
    engine = GameEngine(players, dealer=dealer)
    engine.deal()
    engine.run_auction()
    team_points, _contract = engine.play()
    reward = team_points[0] - team_points[1]  # RL team = seats 0/2, parity 0
    return reward, engine.history['deal_hands']


def counterfactual_reward(hands, dealer, opponent_factory=_default_opponent):
    """Replays exactly the same donne (same hands, same dealer) with
    `opponent_factory` at all 4 seats: a low-noise reference to isolate
    the policy's contribution to play relative to what this opponent
    would have done on the same cards (remarques_rl.md point 3), rather
    than comparing against a global average made noisy by the donne's
    luck. Must always receive the same `opponent_factory` as the
    `run_episode` call that produced `hands` (cf. the training loop in
    `main`): with a pool of opponents, comparing against a different
    opponent than the one actually faced this episode would mix two
    incoherent signals. bid() (Heuristic as well as RLPlayer, which
    inherits it as-is) has no randomness: the replayed contract is
    guaranteed identical to the original donne's."""
    players = [opponent_factory(f'H{i}') for i in range(4)]
    engine = GameEngine(players, dealer=dealer)
    engine.deal(hands=hands)
    engine.run_auction()
    team_points, _contract = engine.play()
    return team_points[0] - team_points[1]


def evaluate(policy, n_games, start_dealer=0, opponent_factory=_default_opponent):
    """Evaluates in greedy mode against `opponent_factory` (HeuristicPlayer
    by default, including during pool training -- cf. `main` -- to keep
    progress tracking comparable across runs): average reward and win
    rate (fraction of donnes where the RL team scores more than the opponent)."""
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
    """Effective entropy coefficient at episode `ep` (>=1) according to
    the chosen decay schedule: 'none' (constant), 'invsqrt'
    (base/sqrt(ep)), or 'inv' (base/ep)."""
    if decay == 'none':
        return base_beta
    if decay == 'invsqrt':
        return base_beta / (ep ** 0.5)
    if decay == 'inv':
        return base_beta / ep
    raise ValueError(f'unknown decay schedule: {decay}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training donnes.')
    parser.add_argument('--eval-every', type=int, default=200, help='Frequency (in episodes) of greedy evaluations.')
    parser.add_argument('--eval-games', type=int, default=100, help='Number of donnes per evaluation.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--entropy-beta', type=float, default=0.01,
                         help="Weight of the entropy bonus in the REINFORCE loss (forces exploration, "
                              "especially useful after a --load of a policy pretrained by imitation).")
    parser.add_argument('--no-counterfactual-baseline', action='store_true',
                         help="Disables the counterfactual (remarques_rl.md point 3): falls back to "
                              "using the raw reward directly, with only NeuralPolicy's EMA baseline.")
    parser.add_argument('--opponent', default='heuristic',
                         help="Opponent(s) at seats 1/3 (and for the counterfactual), comma-separated: "
                              "'heuristic' and/or paths to saved NeuralPolicy weights (self-play "
                              "against a frozen copy, greedy play, never updated). With several values, "
                              "an opponent is sampled at random each episode (a pool rather than a single "
                              "fixed opponent); the eval logged during training always stays vs heuristic "
                              "for comparable tracking across runs.")
    parser.add_argument('--entropy-decay', choices=['none', 'invsqrt', 'inv'], default='none',
                         help="Entropy bonus decay over episodes: 'none' (constant), "
                              "'invsqrt' (beta/sqrt(ep)), 'inv' (beta/ep).")
    parser.add_argument('--pfsp', action='store_true',
                         help="Prioritized Fictitious Self-Play: biases --opponent sampling toward the "
                              "toughest opponents (lowest current win rate) instead of uniform "
                              "sampling (cf. PFSPSampler). No effect if --opponent has only one value.")
    parser.add_argument('--pfsp-refresh-every', type=int, default=5000,
                         help="Frequency (in episodes) of PFSP sampling distribution recomputation.")
    parser.add_argument('--pfsp-temperature', type=float, default=1.0,
                         help="PFSP softmax temperature: lower = stronger bias toward the toughest "
                              "opponent, higher = closer to uniform sampling.")
    parser.add_argument('--pfsp-ema-beta', type=float, default=0.98,
                         help="Exponential moving average coefficient for per-opponent win rate "
                              "(PFSP): closer to 1 = longer memory.")
    parser.add_argument('--pfsp-explore-eps', type=float, default=0.0,
                         help="PFSP exploration floor (0 to 1): mixes the softmax with uniform "
                              "sampling, guaranteeing no opponent falls below explore_eps/N "
                              "probability -- useful to offset an aggressive --pfsp-temperature (cf. "
                              "PFSPSampler). Default 0.0 = no floor, unchanged behavior.")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save', default=None, help="Path to save the weights at the end of training.")
    parser.add_argument('--load', default=None, help='Path to resume from saved weights.')
    parser.add_argument('--checkpoint-every', type=int, default=None,
                         help="Saves a checkpoint every N episodes, in addition to --save at the end of training.")
    parser.add_argument('--checkpoint-dir', default=None,
                         help='Directory to save checkpoints in (required with --checkpoint-every).')
    parser.add_argument('--episode-offset', type=int, default=0,
                         help="Offset added to the episode counter (logs, dealer, entropy decay, "
                              "checkpoint names): to resume a previous training run with consistent "
                              "absolute numbering, set it to the number of episodes already completed.")
    parser.add_argument('--architecture', choices=['small', 'big'], default='small',
                         help="'small' = CardNet (1 hidden layer of 128, default, already-validated "
                              "config). 'big' = CardNetBig (128/128/64, GELU, remarques_rl.md "
                              "point 6) -- --load must then point to a checkpoint generated "
                              "with pretrain.py --architecture big.")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch is required to train a NeuralPolicy (see requirements.txt).")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    net_cls = CardNetBig if args.architecture == 'big' else CardNet
    opponent_pool = build_opponent_pool(args.opponent, net_cls=net_cls)
    sampler = PFSPSampler(opponent_pool, refresh_every=args.pfsp_refresh_every,
                           temperature=args.pfsp_temperature, ema_beta=args.pfsp_ema_beta,
                           explore_eps=args.pfsp_explore_eps) if args.pfsp else None

    policy = NeuralPolicy(lr=args.lr, entropy_beta=args.entropy_beta, net_cls=net_cls)
    if args.load:
        policy.load(args.load)
        print('weights loaded from', args.load)

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

        window.append(reward)  # raw reward, for interpretable tracking (vs the opponent)
        adv_window.append(training_reward)  # signal actually used for the update
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
        print('weights saved to', args.save)

    if sampler is not None:
        print('PFSP picks total:', sampler.summary())


if __name__ == '__main__':
    main()
