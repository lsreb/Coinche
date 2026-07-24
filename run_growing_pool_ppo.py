"""Equivalent PPO de `run_growing_pool.py` (pool grandissant + PFSP), pour
verifier si le gain de polyvalence confirme sur REINFORCE (n=3 seeds, cf.
rl_experiments_v3/README.md) se retrouve aussi avec PPO -- jamais teste
jusqu'ici, `train_ppo.py` n'ayant jamais ete pilote par un orchestrateur en
pool grandissant.

Meme logique exactement que `run_growing_pool.py` : decoupe l'entrainement en
segments de `--segment-episodes` episodes, et apres chaque segment, ajoute le
checkpoint qui vient d'etre produit au pool d'adversaires (--opponent,
train_ppo.py) pour le segment suivant. Chaque segment est un sous-process
`python train_ppo.py` independant (pas d'appel direct a sa fonction main()
dans ce process, meme raison que run_growing_pool.py : eviter toute fuite
d'etat entre segments).

Hyperparametres PPO par defaut : `epochs=8, lr=3e-5` -- la config stable deja
validee sur 3+ seeds contre heuristic seul dans rl_experiments_v3/README.md
(`exp_ppo_lr3e-5`), reprise ici a l'identique (meme --value-coef,
--entropy-coef, --entropy-decay, --clip-eps, --minibatch-size,
--batch-episodes) pour que la seule variable qui change soit "pool
grandissant" vs "heuristic seul", comme pour la version REINFORCE.

`--load-value` : contrairement a REINFORCE (un seul reseau), PPO a un critic
separe (ValueNet). Le tout premier segment part de `--init-load` (imit.pt,
policy seulement, pas de critic pre-entraine -- critic demarre aleatoire).
Les segments suivants rechargent a la fois policy ET critic depuis le
checkpoint du segment precedent (meme fichier passe a --load et --load-value,
cf. train_ppo.py PPOPolicy.save()/load() : un checkpoint PPO natif est un
dict {'policy':, 'value':}).

Usage:
    python run_growing_pool_ppo.py --total-episodes 300000 --segment-episodes 50000 --seed-start 20
    python run_growing_pool_ppo.py --total-episodes 200 --segment-episodes 100  # smoke-test rapide
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
                         help='Poids de depart (policy seule) du tout premier segment.')
    parser.add_argument('--out-dir', default='rl_experiments_v3/exp_growing_pool_ppo',
                         help='Dossier de sortie (checkpoints + logs de chaque segment).')
    parser.add_argument('--batch-episodes', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=8, help='Passes PPO par batch collecte.')
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.002)
    parser.add_argument('--entropy-decay', default='none')
    parser.add_argument('--pfsp-refresh-every', type=int, default=1000)
    parser.add_argument('--pfsp-temperature', type=float, default=0.1)
    parser.add_argument('--pfsp-ema-beta', type=float, default=0.98)
    parser.add_argument('--eval-every', type=int, default=2000)
    parser.add_argument('--eval-games', type=int, default=500)
    parser.add_argument('--checkpoint-every', type=int, default=2000)
    parser.add_argument('--seed-start', type=int, default=20,
                         help="Seed du 1er segment ; incremente de 1 a chaque segment suivant "
                              "(donnes differentes a chaque reprise).")
    parser.add_argument('--start-segment', type=int, default=1,
                         help="Reprendre a partir de ce segment (1-based) si l'orchestration a deja "
                              "tourne partiellement -- suppose que les checkpoints des segments "
                              "precedents existent deja dans --out-dir.")
    args = parser.parse_args()

    if args.total_episodes % args.segment_episodes != 0:
        raise SystemExit("--total-episodes doit etre un multiple de --segment-episodes.")
    os.makedirs(args.out_dir, exist_ok=True)
    n_segments = args.total_episodes // args.segment_episodes

    pool_checkpoints = []  # checkpoints finaux des segments precedents, dans l'ordre
    for seg in range(1, n_segments + 1):
        offset = (seg - 1) * args.segment_episodes
        global_end = seg * args.segment_episodes
        ckpt_path = os.path.join(args.out_dir, f'seg_{global_end}.pt')

        if seg < args.start_segment:
            pool_checkpoints.append(ckpt_path)  # deja fait lors d'une reprise precedente
            continue

        load_path = args.init_load if seg == 1 else os.path.join(args.out_dir, f'seg_{offset}.pt')
        opponent = ','.join(['heuristic'] + pool_checkpoints)
        seed = args.seed_start + seg - 1
        seg_ckpt_dir = os.path.join(args.out_dir, f'seg_{global_end}_checkpoints')

        cmd = [
            sys.executable, 'train_ppo.py',
            '--load', load_path,
            '--opponent', opponent,
            '--pfsp',
            '--pfsp-refresh-every', str(args.pfsp_refresh_every),
            '--pfsp-temperature', str(args.pfsp_temperature),
            '--pfsp-ema-beta', str(args.pfsp_ema_beta),
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
            '--eval-every', str(args.eval_every),
            '--eval-games', str(args.eval_games),
            '--checkpoint-every', str(args.checkpoint_every),
            '--checkpoint-dir', seg_ckpt_dir,
            '--save', ckpt_path,
            '--seed', str(seed),
        ]
        if seg > 1:
            cmd += ['--load-value', load_path]
        log_path = os.path.join(args.out_dir, f'seg_{global_end}.log')
        print(f'=== Segment {seg}/{n_segments} : episodes {offset + 1}-{global_end} '
              f'(pool a {len(pool_checkpoints) + 1} membres) ===', flush=True)
        print('  pool:', opponent, flush=True)
        start = time.perf_counter()
        with open(log_path, 'w') as logf:
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            raise SystemExit(f'Segment {seg} a echoue (code {result.returncode}), voir {log_path}')
        print(f'  segment {seg} termine en {elapsed:.0f}s, poids dans {ckpt_path}', flush=True)

        pool_checkpoints.append(ckpt_path)

    print(f'Orchestration terminee : {n_segments} segments, pool final a {len(pool_checkpoints) + 1} '
          f'membres (heuristic + {len(pool_checkpoints)} snapshots).')


if __name__ == '__main__':
    main()
