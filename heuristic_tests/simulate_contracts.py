"""Simule un batch de donnes et tabule, pour chaque palier de contrat annonce, la
distribution du score reellement marque par l'equipe preneuse (par tranches de 10
points). Sert a verifier si les heuristiques d'encheres sont bien calibrees : un
contrat a 100 devrait le plus souvent aboutir a un score autour de 100+, pas 60.

Usage:
    python heuristic_tests/simulate_contracts.py --games 300
    python heuristic_tests/simulate_contracts.py --games 500 --strategies heuristic,heuristic,heuristic,heuristic --csv out.csv
"""
import argparse
import csv
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coinche.game import GameEngine
from coinche.players import create_player


def ta_points_to_normal_scale(raw_points):
    """A TA il y a 248 points en tout et pas de der, au lieu de 162 : la correspondance
    120-135-150-165-180-195-210-225-240 <-> 80-90-100-110-120-130-140-150-160
    (regles_coinche.md §5) donne 15 points TA pour 10 points normaux. Le capot (250)
    n'est pas concerne par cette conversion. (Copie de play_test.py pour que ce
    script reste autonome.)"""
    if raw_points >= 250:
        return raw_points
    return round(80 + (raw_points - 120) * (10 / 15))


def run_one_game(strategies, dealer):
    players = [create_player(s.strip(), f'P{i}') for i, s in enumerate(strategies)]
    engine = GameEngine(players, dealer=dealer)
    engine.deal()
    engine.run_auction()
    team_points, contract = engine.play()
    taker_team = engine.taker_idx % 2
    points = team_points[taker_team]
    if contract.trump == 'TA':
        points = ta_points_to_normal_scale(points)
    level = 'capot' if contract.capot else contract.level
    return level, contract.trump, points


def build_table(rows):
    counts = defaultdict(lambda: defaultdict(int))
    for level, _trump, points in rows:
        bucket = (points // 10) * 10
        counts[level][bucket] += 1
    buckets = sorted({b for level_counts in counts.values() for b in level_counts})
    level_order = [lvl for lvl in (80, 90, 100, 110, 120, 130, 140, 150, 160) if lvl in counts]
    if 'capot' in counts:
        level_order.append('capot')
    return counts, buckets, level_order


def print_table(rows):
    counts, buckets, level_order = build_table(rows)
    header = ['contrat'] + [str(b) for b in buckets] + ['total']
    col_w = max(6, max(len(str(b)) for b in buckets) + 1) if buckets else 6
    print(' | '.join(f'{h:>{col_w}}' for h in header))
    print('-' * (len(header) * (col_w + 3)))
    for level in level_order:
        level_counts = counts[level]
        total = sum(level_counts.values())
        made = sum(v for b, v in level_counts.items() if isinstance(level, int) and b >= level) if isinstance(level, int) else total
        row = [str(level)] + [str(level_counts.get(b, 0)) for b in buckets] + [f'{total} ({100*made//total}% reussi)' if total else '0']
        print(' | '.join(f'{c:>{col_w}}' for c in row))


def main():
    parser = argparse.ArgumentParser(description='Simule des donnes et tabule le score par contrat.')
    parser.add_argument('--games', type=int, default=300, help='Nombre de donnes a simuler.')
    parser.add_argument('--strategies', default='heuristic,heuristic,heuristic,heuristic',
                         help='Comma-separated strategies for seats 0-3 (voir play_test.py).')
    parser.add_argument('--seed', type=int, default=None, help='Graine aleatoire optionnelle (reproductibilite).')
    parser.add_argument('--csv', default=None, help='Chemin optionnel pour sauvegarder le detail brut par donne.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    strategies = args.strategies.split(',')
    rows = []
    for i in range(args.games):
        rows.append(run_one_game(strategies, dealer=i % 4))

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['level', 'trump', 'points'])
            w.writerows(rows)
        print(f'Detail brut ({len(rows)} donnes) sauvegarde dans {args.csv}')

    print_table(rows)


if __name__ == '__main__':
    main()
