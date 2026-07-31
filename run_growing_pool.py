"""Orchestre un entrainement REINFORCE en pool grandissant (remarques_rl.md,
discussion "pool + PFSP") : decoupe l'entrainement en segments de
`--segment-episodes` episodes, et apres chaque segment, ajoute le checkpoint
qui vient d'etre produit au pool d'adversaires (--opponent, train.py) pour le
segment suivant -- au lieu d'un pool fixe (exp_pool_350k) ou d'un adversaire
fige unique (exp_selfplay_frozen100k/150k), qui plafonnaient tous les deux une
fois "battus en moyenne" par la policy en cours d'entrainement.

Chaque segment est un sous-process `python train.py` independant (pas d'appel
direct a train.main() dans ce process, pour eviter tout risque de fuite
d'etat entre segments). Le pool de chaque segment est "heuristic" + tous les
checkpoints finaux des segments precedents, et PFSP (--pfsp) est toujours
active pour biaiser le tirage vers l'adversaire du pool actuellement le plus
coriace (sans effet le tout premier segment, ou le pool ne contient que
"heuristic").

Part par defaut de `imit.pt`, pas d'un ancien checkpoint exp_* : la regle 5
du scoring (chute, arrondi dizaine, bonus capot, echelle TA) vient d'etre
corrigee dans coinche/game.py, donc tous les checkpoints RL anterieurs ont
appris contre l'ancien signal simplifie -- on repart de zero pour integrer le
nouveau signal des le debut plutot que de continuer sur un biais herite.

Usage:
    python run_growing_pool.py --total-episodes 500000 --segment-episodes 50000
    python run_growing_pool.py --total-episodes 200 --segment-episodes 100  # smoke-test rapide
"""
import argparse
import os
import re
import subprocess
import sys
import time


def _parse_final_winrates(log_path):
    """Parse la derniere ligne 'PFSP picks total: ...' d'un log de segment
    (imprimee par train.py/PFSPSampler.summary()) pour recuperer le win-rate
    final de la policy contre chaque adversaire de CE segment -- utilise pour
    elaguer le pool par faiblesse (cf. --pool-max-size). Chaque segment est un
    sous-process frais (PFSPSampler.ema_winrate redemarre a 0.5 pour tout le
    monde a chaque invocation de train.py), donc ce resume final logge est la
    seule trace disponible du win-rate d'un segment a l'autre."""
    with open(log_path) as f:
        lines = f.readlines()
    summary_line = next((l for l in reversed(lines) if l.startswith('PFSP picks total:')), None)
    if summary_line is None:
        return {}
    return {name: float(wr) for name, wr in re.findall(r'(\S+):\d+x\(wr=([\d.]+)\)', summary_line)}


def _prune_pool(pool_checkpoints, keep_every):
    """Garde un checkpoint sur `keep_every` (par index de segment, 1-based),
    plus toujours le plus recent -- reduit la dilution du budget PFSP par
    adversaire a mesure que le pool grandit (mesure empiriquement le
    2026-07-30 : le % de parties jouees contre le dernier ajoute est passe de
    ~15% a ~8% entre 300k et 750k, en meme temps que le gain de force par
    palier de 150k s'est effondre de ~+5 a +2). Le plus recent est toujours
    garde meme s'il ne tombe pas sur le multiple, car c'est justement
    l'adversaire le plus pertinent a apprendre a battre."""
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
                         help='Poids de depart du tout premier segment.')
    parser.add_argument('--out-dir', default='rl_experiments_v2/exp_growing_pool',
                         help='Dossier de sortie (checkpoints + logs de chaque segment). Dans '
                              'rl_experiments_v2/ (pas rl_experiments/) : la regle 5 du scoring '
                              '(chute/arrondi/capot/echelle TA) vient de changer, donc cette lignee '
                              "n'est plus comparable a celle d'avant (imit.pt reste le seul point "
                              'commun entre les deux, son reward ne dependant pas du scoring final).')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--entropy-beta', type=float, default=0.002)
    parser.add_argument('--entropy-decay', default='none')
    parser.add_argument('--architecture', choices=['small', 'big'], default='small',
                         help="Passe directement a train.py --architecture (meme choix/semantique) : "
                              "'big' pour partir d'un checkpoint CardNetBig (ex. "
                              "rl_experiments_v4/imit_bignet_ent02.pt) -- --init-load et tous les "
                              "checkpoints de segments generes doivent alors etre de cette meme "
                              "architecture (le pool d'adversaires en depend aussi, cf. "
                              "train.py build_opponent_pool).")
    parser.add_argument('--pfsp-refresh-every', type=int, default=1000)
    parser.add_argument('--pfsp-temperature', type=float, default=0.05,
                         help="Discussion du 2026-07-30 : baisse de 0.1 a 0.05 pour biaiser plus fort vers "
                              "l'adversaire le plus coriace et accelerer l'apprentissage contre lui -- "
                              "combiner avec --pfsp-explore-eps (sinon un adversaire facile peut tomber a "
                              "moins de 1% des parties, calcule le meme jour).")
    parser.add_argument('--pfsp-ema-beta', type=float, default=0.98)
    parser.add_argument('--pfsp-explore-eps', type=float, default=0.01,
                         help="Plancher d'exploration PFSP (cf. train.py PFSPSampler) : garantit qu'aucun "
                              "adversaire ne tombe sous explore_eps/N de probabilite, compagnon obligatoire "
                              "d'une --pfsp-temperature agressive comme 0.05 (defaut ici).")
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
    parser.add_argument('--pool-keep-every', type=int, default=1,
                         help="Ne garde dans le pool d'ADVERSAIRES (--opponent de train.py) qu'un "
                              "checkpoint sur N par index de segment, plus toujours le plus recent -- "
                              "ex. 2 = multiples de 2*segment-episodes (typiquement 100k si "
                              "segment-episodes=50000). Tous les checkpoints continuent d'etre generes "
                              "et sauvegardes normalement (--out-dir garde l'historique complet) ; seul "
                              "le sous-ensemble propose comme adversaire est reduit, pour limiter la "
                              "dilution du budget PFSP par adversaire a mesure que le pool grandit "
                              "(defaut 1 = tout garder, comportement inchange).")
    parser.add_argument('--pool-max-size', type=int, default=None,
                         help="Plafond dur sur le nombre d'adversaires (hors heuristic, toujours en "
                              "plus) proposes a train.py. Applique APRES --pool-keep-every : si le pool "
                              "encore trop grand, retire l'adversaire au win-rate le plus haut (le moins "
                              "coriace, mesure via le resume PFSP final du segment qui vient de tourner, "
                              "cf. _parse_final_winrates) -- jamais le tout dernier ajoute. PAS un bannissement "
                              "definitif : reevalue a chaque segment a partir des donnees les plus fraiches, "
                              "donc un adversaire ecarte peut revenir plus tard s'il n'y a pas de donnee de "
                              "win-rate plus recente et defavorable sur les autres (fallback neutre 0.5, cf. "
                              "_parse_final_winrates) -- le checkpoint lui-meme n'est jamais supprime du "
                              "disque, seulement exclu ou reintegre comme adversaire propose. Defaut aucun "
                              "plafond (comportement inchange).")
    args = parser.parse_args()

    if args.total_episodes % args.segment_episodes != 0:
        raise SystemExit("--total-episodes doit etre un multiple de --segment-episodes.")
    os.makedirs(args.out_dir, exist_ok=True)
    n_segments = args.total_episodes // args.segment_episodes

    pool_checkpoints = []  # checkpoints finaux des segments precedents, dans l'ordre
    prev_log_path = None  # log du dernier segment reellement tourne (pour le win-rate final, cf. --pool-max-size)
    for seg in range(1, n_segments + 1):
        offset = (seg - 1) * args.segment_episodes
        global_end = seg * args.segment_episodes
        ckpt_path = os.path.join(args.out_dir, f'seg_{global_end}.pt')

        if seg < args.start_segment:
            pool_checkpoints.append(ckpt_path)  # deja fait lors d'une reprise precedente
            prev_log_path = os.path.join(args.out_dir, f'seg_{global_end}.log')
            continue

        load_path = args.init_load if seg == 1 else os.path.join(args.out_dir, f'seg_{offset}.pt')
        candidate_pool = _prune_pool(pool_checkpoints, args.pool_keep_every)
        if args.pool_max_size and len(candidate_pool) > args.pool_max_size and prev_log_path:
            # Elague par faiblesse (win-rate le plus haut = le moins coriace, mesure lors du
            # dernier segment reellement tourne) jusqu'au plafond -- jamais le tout dernier
            # ajoute (candidate_pool[-1], cf. _prune_pool) qui reste toujours garde.
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
        print(f'=== Segment {seg}/{n_segments} : episodes {offset + 1}-{global_end} '
              f'(pool a {len(pruned_pool) + 1} membres, {len(pool_checkpoints)} disponibles) ===', flush=True)
        print('  pool:', opponent, flush=True)
        start = time.perf_counter()
        with open(log_path, 'w') as logf:
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            raise SystemExit(f'Segment {seg} a echoue (code {result.returncode}), voir {log_path}')
        print(f'  segment {seg} termine en {elapsed:.0f}s, poids dans {ckpt_path}', flush=True)

        pool_checkpoints.append(ckpt_path)
        prev_log_path = log_path

    print(f'Orchestration terminee : {n_segments} segments, pool final a {len(pool_checkpoints) + 1} '
          f'membres (heuristic + {len(pool_checkpoints)} snapshots).')


if __name__ == '__main__':
    main()
