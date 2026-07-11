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
import time
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
    # Points de plis bruts (independants de la coinche, cf. GameEngine.play) :
    # sert au tableau de distribution, sur l'echelle habituelle (comme si le
    # contrat n'etait pas coinche), pour la calibration des encheres.
    raw_points = engine.history['raw_points'][taker_team]
    if contract.trump == 'TA':
        raw_points = ta_points_to_normal_scale(raw_points)
    # Score reellement marque (double si coinche) : sert au calcul de la moyenne.
    real_points = team_points[taker_team]
    level = 'capot' if contract.capot else contract.level
    return level, contract.trump, raw_points, contract.coinched, real_points


def trump_category(trump):
    """Regroupe les contrats en 3 familles : couleur (P/C/K/T), SA, TA."""
    return trump if trump in ('SA', 'TA') else 'Couleur'


def split_by_category(rows):
    cats = defaultdict(list)
    for row in rows:
        cats[trump_category(row[1])].append(row)
    return cats


def build_table(rows):
    counts = defaultdict(lambda: defaultdict(int))
    coinched_count = defaultdict(int)
    real_sum = defaultdict(float)
    for level, _trump, points, coinched, real_points in rows:
        bucket = (points // 10) * 10
        counts[level][bucket] += 1
        if coinched:
            coinched_count[level] += 1
        real_sum[level] += real_points
    buckets = sorted({b for level_counts in counts.values() for b in level_counts})
    level_order = [lvl for lvl in (80, 90, 100, 110, 120, 130, 140, 150, 160) if lvl in counts]
    if 'capot' in counts:
        level_order.append('capot')
    return counts, buckets, level_order, coinched_count, real_sum


def print_table(rows):
    counts, buckets, level_order, coinched_count, real_sum = build_table(rows)
    header = ['contrat'] + [str(b) for b in buckets] + ['total', '% coinche', 'moyenne']
    col_w = max(6, max(len(str(b)) for b in buckets) + 1) if buckets else 6
    print(' | '.join(f'{h:>{col_w}}' for h in header))
    print('-' * (len(header) * (col_w + 3)))
    for level in level_order:
        level_counts = counts[level]
        total = sum(level_counts.values())
        # Un capot annonce n'est reussi que si le score atteint reellement 250 (+20
        # belote), pas par defaut : un capot chute retombe sur le score de plis normal.
        ref = 250 if level == 'capot' else level
        made = sum(v for b, v in level_counts.items() if b >= ref)
        pct_coinche = 100 * coinched_count[level] / total if total else 0
        # Moyenne sur le score reellement marque (donc double en cas de coinche),
        # contrairement aux colonnes de distribution ci-dessus qui restent sur
        # l'echelle brute habituelle, comme si aucune donne n'etait coinchee.
        moyenne = real_sum[level] / total if total else 0
        row = ([str(level)] + [str(level_counts.get(b, 0)) for b in buckets]
               + [f'{total} ({100*made//total}% reussi)' if total else '0',
                  f'{pct_coinche:.0f}%',
                  f'{moyenne:.0f}'])
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
    start = time.perf_counter()
    for i in range(args.games):
        rows.append(run_one_game(strategies, dealer=i % 4))
    elapsed = time.perf_counter() - start
    print(f'{len(rows)} donnes simulees en {elapsed:.2f}s ({1000 * elapsed / len(rows):.1f} ms/donne)')

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['level', 'trump', 'raw_points', 'coinched', 'real_points'])
            w.writerows(rows)
        print(f'Detail brut ({len(rows)} donnes) sauvegarde dans {args.csv}')

    cats = split_by_category(rows)
    for cat_name in ('Couleur', 'SA', 'TA'):
        cat_rows = cats.get(cat_name, [])
        if not cat_rows:
            continue
        print(f'\n=== {cat_name} ({len(cat_rows)} donnes) ===')
        print_table(cat_rows)


if __name__ == '__main__':
    main()
