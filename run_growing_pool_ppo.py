"""PPO equivalent of `run_growing_pool.py` (growing pool + PFSP), to check
whether the versatility gain confirmed on REINFORCE (n=3 seeds, cf.
rl_experiments_v3/README.md) also shows up with PPO -- never tested before
now, `train_ppo.py` having never been driven by a growing-pool orchestrator.

Exactly the same logic as `run_growing_pool.py`: splits training into
segments of `--segment-episodes` episodes, and after each segment, adds the
checkpoint just produced to the opponent pool (--opponent, train_ppo.py) for
the next segment. Each segment is an independent `python train_ppo.py`
subprocess (no direct call to its main() function in this process, same
reason as run_growing_pool.py: avoid any state leaking between segments).

Default PPO hyperparameters: `epochs=8, lr=3e-5` -- the stable config
already validated over 3+ seeds against heuristic alone in
rl_experiments_v3/README.md (`exp_ppo_lr3e-5`), reused here identically
(same --value-coef, --entropy-coef, --entropy-decay, --clip-eps,
--minibatch-size, --batch-episodes) so the only variable that changes is
"growing pool" vs "heuristic alone", as for the REINFORCE version.

`--load-value`: for `--architecture small/big`, unlike REINFORCE (a single
network), PPO has a separate critic (ValueNet). The very first segment
starts from `--init-load` (imit.pt, policy only, no pretrained critic --
critic starts out random). The following segments reload both policy AND
critic from the previous segment's checkpoint (the same file passed to
--load and --load-value, cf. train_ppo.py PPOPolicy.save()/load(): a native
PPO checkpoint is a dict {'policy':, 'value':}). For `--architecture
shared/shared_aux`, a single network (trunk+actor+critic[+aux]) is loaded
via --load alone -- --load-value is never passed (train_ppo.py refuses it
for these architectures).

Usage:
    python run_growing_pool_ppo.py --total-episodes 300000 --segment-episodes 50000 --seed-start 20
    python run_growing_pool_ppo.py --total-episodes 200 --segment-episodes 100  # quick smoke test
"""
import argparse
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-episodes', type=int, default=300000)
    parser.add_argument('--segment-episodes', type=int, default=50000)
    parser.add_argument('--init-load', default='rl_experiments_v3/imit.pt',
                         help='Starting weights (policy only) for the very first segment.')
    parser.add_argument('--out-dir', default='rl_experiments_v3/exp_growing_pool_ppo',
                         help='Output directory (each segment\'s checkpoints + logs).')
    parser.add_argument('--batch-episodes', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=8, help='PPO passes per collected batch.')
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.002)
    parser.add_argument('--entropy-decay', default='none')
    parser.add_argument('--architecture', choices=['small', 'big', 'shared', 'shared_aux'], default='small',
                         help="Passed straight through to train_ppo.py --architecture (same "
                              "choices/semantics): 'big' to start from a CardNetBig checkpoint "
                              "(e.g. rl_experiments_v4/imit_bignet_ent02.pt), 'shared'/'shared_aux' "
                              "for a shared actor/critic trunk (e.g. from a checkpoint already "
                              "converted via load_actor_from_cardnetbig) -- --init-load and all "
                              "generated segment checkpoints must then be of this same architecture.")
    parser.add_argument('--aux-coef', type=float, default=0.5,
                         help="Weight of the auxiliary loss -- only with --architecture shared_aux "
                              "(passed as-is to train_ppo.py --aux-coef).")
    parser.add_argument('--pfsp-refresh-every', type=int, default=1000)
    parser.add_argument('--pfsp-temperature', type=float, default=0.1)
    parser.add_argument('--pfsp-ema-beta', type=float, default=0.98)
    parser.add_argument('--pfsp-explore-eps', type=float, default=0.0,
                         help="PFSP exploration floor (cf. train.py PFSPSampler, 2026-07-30 "
                              "discussion): guarantees no opponent falls below explore_eps/N "
                              "probability -- recommended companion to an aggressive --pfsp-temperature.")
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
    args = parser.parse_args()

    if args.total_episodes % args.segment_episodes != 0:
        raise SystemExit("--total-episodes must be a multiple of --segment-episodes.")
    os.makedirs(args.out_dir, exist_ok=True)
    n_segments = args.total_episodes // args.segment_episodes

    pool_checkpoints = []  # previous segments' final checkpoints, in order
    for seg in range(1, n_segments + 1):
        offset = (seg - 1) * args.segment_episodes
        global_end = seg * args.segment_episodes
        ckpt_path = os.path.join(args.out_dir, f'seg_{global_end}.pt')

        if seg < args.start_segment:
            pool_checkpoints.append(ckpt_path)  # already done in a previous resume
            continue

        load_path = args.init_load if seg == 1 else os.path.join(args.out_dir, f'seg_{offset}.pt')
        opponent = ','.join(['heuristic'] + pool_checkpoints)
        seed = args.seed_start + seg - 1
        seg_ckpt_dir = os.path.join(args.out_dir, f'seg_{global_end}_checkpoints')

        cmd = [
            sys.executable, '-u', 'train_ppo.py',
            '--load', load_path,
            '--opponent', opponent,
            '--pfsp',
            '--pfsp-refresh-every', str(args.pfsp_refresh_every),
            '--pfsp-temperature', str(args.pfsp_temperature),
            '--pfsp-ema-beta', str(args.pfsp_ema_beta),
            '--pfsp-explore-eps', str(args.pfsp_explore_eps),
            '--episodes', str(args.segment_episodes),
            '--episode-offset', str(offset),
            '--batch-episodes', str(args.batch_episodes),
            '--epochs', str(args.epochs),
            '--minibatch-size', str(args.minibatch_size),
            '--clip-eps', str(args.clip_eps),
            '--lr', str(args.lr),
            '--value-coef', str(args.value_coef),
            '--entropy-coef', str(args.entropy_coef),
            '--entropy-decay', args.entropy_decay,
            '--architecture', args.architecture,
        ]
        if args.architecture == 'shared_aux':
            cmd += ['--aux-coef', str(args.aux_coef)]
        cmd += [
            '--eval-every', str(args.eval_every),
            '--eval-games', str(args.eval_games),
            '--checkpoint-every', str(args.checkpoint_every),
            '--checkpoint-dir', seg_ckpt_dir,
            '--save', ckpt_path,
            '--seed', str(seed),
        ]
        if seg > 1 and args.architecture not in ('shared', 'shared_aux'):
            # --load-value doesn't make sense for shared/shared_aux (a single network
            # loaded by --load, no separate value_net, cf. train_ppo.py main()).
            cmd += ['--load-value', load_path]
        log_path = os.path.join(args.out_dir, f'seg_{global_end}.log')
        print(f'=== Segment {seg}/{n_segments}: episodes {offset + 1}-{global_end} '
              f'(pool has {len(pool_checkpoints) + 1} members) ===', flush=True)
        print('  pool:', opponent, flush=True)
        start = time.perf_counter()
        with open(log_path, 'w') as logf:
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            raise SystemExit(f'Segment {seg} failed (code {result.returncode}), see {log_path}')
        print(f'  segment {seg} finished in {elapsed:.0f}s, weights in {ckpt_path}', flush=True)

        pool_checkpoints.append(ckpt_path)

    print(f'Orchestration finished: {n_segments} segments, final pool has {len(pool_checkpoints) + 1} '
          f'members (heuristic + {len(pool_checkpoints)} snapshots).')


if __name__ == '__main__':
    main()
