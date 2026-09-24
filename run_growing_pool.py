"""Orchestrates a growing-pool REINFORCE training run (remarques_rl.md,
"pool + PFSP" discussion): splits training into segments of
`--segment-episodes` episodes, and after each segment, adds the checkpoint
just produced to the opponent pool (--opponent, train.py) for the next
segment -- instead of a fixed pool (exp_pool_350k) or a single frozen
opponent (exp_selfplay_frozen100k/150k), both of which plateaued once
"beaten on average" by the policy being trained.

Each segment is an independent `python train.py` subprocess (no direct call
to train.main() in this process, to avoid any risk of state leaking between
segments). Each segment's pool is "heuristic" + all previous segments' final
checkpoints, and PFSP (--pfsp) is always enabled to bias sampling toward the
currently toughest pool opponent (no effect on the very first segment, where
the pool only contains "heuristic").

Starts from `imit.pt` by default, not an old exp_* checkpoint: scoring rule 5
(falling short, ten-rounding, capot bonus, TA scale) was just fixed in
coinche/game.py, so every prior RL checkpoint learned against the old
simplified signal -- starting from scratch integrates the new signal from
the start rather than continuing on an inherited bias.

Usage:
    python run_growing_pool.py --total-episodes 500000 --segment-episodes 50000
    python run_growing_pool.py --total-episodes 200 --segment-episodes 100  # quick smoke test
"""
import argparse
import os
import re
import subprocess
import sys
import time


def _parse_final_winrates(log_path):
    """Parses a segment log's last 'PFSP picks total: ...' line (printed by
    train.py/PFSPSampler.summary()) to recover the policy's final win rate
    against each opponent of THIS segment -- used to prune the pool by
    weakness (cf. --pool-max-size). Each segment is a fresh subprocess
    (PFSPSampler.ema_winrate restarts at 0.5 for everyone on every train.py
    invocation), so this logged final summary is the only trace of a
    segment's win rate available across segments."""
    with open(log_path) as f:
        lines = f.readlines()
    summary_line = next((l for l in reversed(lines) if l.startswith('PFSP picks total:')), None)
    if summary_line is None:
        return {}
    return {name: float(wr) for name, wr in re.findall(r'(\S+):\d+x\(wr=([\d.]+)\)', summary_line)}


def _prune_pool(pool_checkpoints, keep_every):
    """Keeps 1 checkpoint out of every `keep_every` (by segment index,
    1-based), plus always the most recent -- reduces the dilution of the
    PFSP budget per opponent as the pool grows (measured empirically on
    2026-07-30: the % of games played against the latest addition dropped
    from ~15% to ~8% between 300k and 750k, at the same time as the
    strength gain per 150k step collapsed from ~+5 to +2). The most recent
    is always kept even if it doesn't land on the multiple, since it's
    precisely the most relevant opponent to learn to beat."""
    if keep_every <= 1:
        return list(pool_checkpoints)
    pruned = [ckpt for i, ckpt in enumerate(pool_checkpoints, start=1) if i % keep_every == 0]
    if pool_checkpoints and pool_checkpoints[-1] not in pruned:
        pruned.append(pool_checkpoints[-1])
    return pruned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-episodes', type=int, default=500000)
    parser.add_argument('--segment-episodes', type=int, default=50000)
    parser.add_argument('--init-load', default='rl_experiments/imit.pt',
                         help='Starting weights for the very first segment.')
    parser.add_argument('--out-dir', default='rl_experiments_v2/exp_growing_pool',
                         help='Output directory (each segment\'s checkpoints + logs). Under '
                              'rl_experiments_v2/ (not rl_experiments/): scoring rule 5 '
                              '(falling short/rounding/capot/TA scale) just changed, so this '
                              "lineage isn't comparable to the earlier one (imit.pt is the only "
                              'point in common between the two, since its reward does not depend on final scoring).')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--entropy-beta', type=float, default=0.002)
    parser.add_argument('--entropy-decay', default='none')
    parser.add_argument('--architecture', choices=['small', 'big'], default='small',
                         help="Passed straight through to train.py --architecture (same choices/"
                              "semantics): 'big' to start from a CardNetBig checkpoint (e.g. "
                              "rl_experiments_v4/imit_bignet_ent02.pt) -- --init-load and all "
                              "generated segment checkpoints must then be of this same "
                              "architecture (the opponent pool depends on it too, cf. "
                              "train.py build_opponent_pool).")
    parser.add_argument('--pfsp-refresh-every', type=int, default=1000)
    parser.add_argument('--pfsp-temperature', type=float, default=0.05,
                         help="2026-07-30 discussion: lowered from 0.1 to 0.05 to bias more "
                              "strongly toward the toughest opponent and speed up learning "
                              "against it -- pair with --pfsp-explore-eps (otherwise an easy "
                              "opponent can drop below 1% of games, computed the same day).")
    parser.add_argument('--pfsp-ema-beta', type=float, default=0.98)
    parser.add_argument('--pfsp-explore-eps', type=float, default=0.01,
                         help="PFSP exploration floor (cf. train.py PFSPSampler): guarantees no "
                              "opponent falls below explore_eps/N probability, a mandatory "
                              "companion to an aggressive --pfsp-temperature like 0.05 (default here).")
    parser.add_argument('--eval-every', type=int, default=2000)
    parser.add_argument('--eval-games', type=int, default=500)
    parser.add_argument('--checkpoint-every', type=int, default=2000)
    parser.add_argument('--seed-start', type=int, default=20,
                         help="Seed for segment 1; incremented by 1 for each following segment "
                              "(different donnes on each resume).")
    parser.add_argument('--start-segment', type=int, default=1,
                         help="Resume from this segment (1-based) if the orchestration already "
                              "ran partially -- assumes previous segments' checkpoints already "
                              "exist in --out-dir.")
    parser.add_argument('--pool-keep-every', type=int, default=1,
                         help="Keeps only 1 checkpoint out of every N (by segment index) in the "
                              "OPPONENT pool (train.py's --opponent), plus always the most recent -- "
                              "e.g. 2 = multiples of 2*segment-episodes (typically 100k if "
                              "segment-episodes=50000). Every checkpoint keeps being generated "
                              "and saved normally (--out-dir keeps the full history); only the "
                              "subset offered as an opponent is reduced, to limit the dilution "
                              "of the PFSP budget per opponent as the pool grows "
                              "(default 1 = keep everything, unchanged behavior).")
    parser.add_argument('--pool-max-size', type=int, default=None,
                         help="Hard cap on the number of opponents (excluding heuristic, always "
                              "added on top) offered to train.py. Applied AFTER --pool-keep-every: "
                              "if the pool is still too big, drops the opponent with the highest "
                              "win rate (the least tough, measured from the final PFSP summary of "
                              "the segment that just ran, cf. _parse_final_winrates) -- never the "
                              "very last one added. NOT a permanent ban: reevaluated at every "
                              "segment from the freshest data, so a dropped opponent can come back "
                              "later if there's no more recent, unfavorable win-rate data on the "
                              "others (neutral 0.5 fallback, cf. _parse_final_winrates) -- the "
                              "checkpoint itself is never deleted from disk, only excluded or "
                              "reincluded as an offered opponent. Default is no cap (unchanged behavior).")
    parser.add_argument('--pool-exclude', default='',
                         help="Comma-separated episode counts (e.g. '50000,150000,250000') to "
                              "MANUALLY and PERMANENTLY exclude from the opponent pool -- translated "
                              "into seg_<N>.pt paths relative to --out-dir. Unlike --pool-max-size "
                              "(adaptive win-rate-based pruning, never permanent), an exclusion "
                              "listed here stays removed for the rest of the orchestration, "
                              "regardless of any win rate measured afterward. Applied before "
                              "--pool-keep-every/--pool-max-size.")
    args = parser.parse_args()

    if args.total_episodes % args.segment_episodes != 0:
        raise SystemExit("--total-episodes must be a multiple of --segment-episodes.")
    os.makedirs(args.out_dir, exist_ok=True)
    n_segments = args.total_episodes // args.segment_episodes

    excluded = {os.path.join(args.out_dir, f'seg_{int(e.strip())}.pt')
                for e in args.pool_exclude.split(',') if e.strip()}

    pool_checkpoints = []  # previous segments' final checkpoints, in order
    prev_log_path = None  # log of the last segment actually run (for the final win rate, cf. --pool-max-size)
    for seg in range(1, n_segments + 1):
        offset = (seg - 1) * args.segment_episodes
        global_end = seg * args.segment_episodes
        ckpt_path = os.path.join(args.out_dir, f'seg_{global_end}.pt')

        if seg < args.start_segment:
            pool_checkpoints.append(ckpt_path)  # already done in a previous resume
            prev_log_path = os.path.join(args.out_dir, f'seg_{global_end}.log')
            continue

        load_path = args.init_load if seg == 1 else os.path.join(args.out_dir, f'seg_{offset}.pt')
        candidate_pool = [c for c in pool_checkpoints if c not in excluded]
        candidate_pool = _prune_pool(candidate_pool, args.pool_keep_every)
        if args.pool_max_size and len(candidate_pool) > args.pool_max_size and prev_log_path:
            # Prunes by weakness (highest win rate = least tough, measured during the
            # last segment actually run) down to the cap -- never the very last one
            # added (candidate_pool[-1], cf. _prune_pool), which is always kept.
            winrates = _parse_final_winrates(prev_log_path)
            most_recent = candidate_pool[-1]
            removable = sorted((c for c in candidate_pool if c != most_recent),
                                key=lambda c: winrates.get(c, 0.5), reverse=True)
            n_to_drop = len(candidate_pool) - args.pool_max_size
            dropped = set(removable[:n_to_drop])
            candidate_pool = [c for c in candidate_pool if c not in dropped]
        pruned_pool = candidate_pool
        opponent = ','.join(['heuristic'] + pruned_pool)
        seed = args.seed_start + seg - 1
        seg_ckpt_dir = os.path.join(args.out_dir, f'seg_{global_end}_checkpoints')

        cmd = [
            sys.executable, '-u', 'train.py',
            '--load', load_path,
            '--opponent', opponent,
            '--pfsp',
            '--pfsp-refresh-every', str(args.pfsp_refresh_every),
            '--pfsp-temperature', str(args.pfsp_temperature),
            '--pfsp-ema-beta', str(args.pfsp_ema_beta),
            '--pfsp-explore-eps', str(args.pfsp_explore_eps),
            '--episodes', str(args.segment_episodes),
            '--episode-offset', str(offset),
            '--lr', str(args.lr),
            '--entropy-beta', str(args.entropy_beta),
            '--entropy-decay', args.entropy_decay,
            '--architecture', args.architecture,
            '--eval-every', str(args.eval_every),
            '--eval-games', str(args.eval_games),
            '--checkpoint-every', str(args.checkpoint_every),
            '--checkpoint-dir', seg_ckpt_dir,
            '--save', ckpt_path,
            '--seed', str(seed),
        ]
        log_path = os.path.join(args.out_dir, f'seg_{global_end}.log')
        print(f'=== Segment {seg}/{n_segments}: episodes {offset + 1}-{global_end} '
              f'(pool has {len(pruned_pool) + 1} members, {len(pool_checkpoints)} available) ===', flush=True)
        print('  pool:', opponent, flush=True)
        start = time.perf_counter()
        with open(log_path, 'w') as logf:
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            raise SystemExit(f'Segment {seg} failed (code {result.returncode}), see {log_path}')
        print(f'  segment {seg} finished in {elapsed:.0f}s, weights in {ckpt_path}', flush=True)

        pool_checkpoints.append(ckpt_path)
        prev_log_path = log_path

    print(f'Orchestration finished: {n_segments} segments, final pool has {len(pool_checkpoints) + 1} '
          f'members (heuristic + {len(pool_checkpoints)} snapshots).')


if __name__ == '__main__':
    main()
