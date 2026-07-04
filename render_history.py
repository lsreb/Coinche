import argparse
import json
from pathlib import Path

from coinche.game import SUITS, TRUMP_ORDER, NORMAL_ORDER, TRUMP_POINTS, NORMAL_POINTS, SA_POINTS

PLAYER_POSITIONS = {0: 'North', 1: 'West', 2: 'South', 3: 'East'}
TEAM_LABELS = {0: 'Nord-Sud', 1: 'Ouest-Est'}

CARD_BG = {
    'P': '#0055aa',
    'C': '#aa0000',
    'K': '#ffaa00',
    'T': '#008800',
}


def card_html(card):
    if len(card) < 2:
        return card
    suit = card[-1]
    rank = card[:-1]
    color = CARD_BG.get(suit, '#333')
    return f'<span class="card" style="background:{color};">{rank}{suit}</span>'


def sort_hand(cards, trump):
    """Trie les cartes par couleur (ordre fixe P/C/K/T), puis dans chaque couleur
    selon l'ordre de force réel (ordre atout si TA ou si c'est la couleur d'atout,
    ordre normal sinon)."""
    def key(card):
        if len(card) < 2:
            return (99, 99)
        suit = card[-1]
        rank = card[:-1]
        order = TRUMP_ORDER if (trump == 'TA' or suit == trump) else NORMAL_ORDER
        suit_idx = SUITS.index(suit) if suit in SUITS else 99
        rank_idx = order.index(rank) if rank in order else 99
        return (suit_idx, rank_idx)
    return sorted(cards, key=key)


def hand_html(cards):
    if not cards:
        return '<span class="empty">(vide)</span>'
    return ' '.join(card_html(c) for c in cards)


def card_point(card, trump):
    if len(card) < 2:
        return 0
    suit = card[-1]
    rank = card[:-1]
    if trump == 'TA' or suit == trump:
        return TRUMP_POINTS.get(rank, 0)
    elif trump == 'SA':
        return SA_POINTS.get(rank, 0)
    return NORMAL_POINTS.get(rank, 0)


def running_team_scores(history, trump):
    """Score cumulé (points de cartes + belote + 10 de der) après chaque pli,
    répliquant exactement la logique de GameEngine.play() (game.py).

    La belote ne compte que si un même joueur détient le Roi ET la Dame
    d'atout dans sa main initiale ; elle est créditée dès que ce joueur a
    joué la seconde des deux cartes, quel que soit le vainqueur du pli."""
    belote_seat = None
    if trump not in ('SA', 'TA'):
        for seat, hand in enumerate(history['deal_hands']):
            ranks = {c[:-1] for c in hand if c[-1] == trump}
            if 'K' in ranks and 'Q' in ranks:
                belote_seat = seat
                break
    belote_played = 0
    belote_awarded = False
    belote_awarded_trick = None
    team_points = {0: 0, 1: 0}
    running = []
    tricks = history['tricks']
    winners = [t.get('winner') for t in tricks]
    capot_team = None
    if len(tricks) == 8 and all(w is not None for w in winners) and len({w % 2 for w in winners}) == 1:
        capot_team = winners[0] % 2
    for idx, trick in enumerate(tricks, start=1):
        winner = trick.get('winner')
        cards = [p['card'] for p in trick['plays']]
        if winner is not None:
            team = winner % 2
            is_last = idx == len(tricks)
            if capot_team is not None and is_last:
                # Une équipe qui fait tous les plis marque 250 (regles_coinche.md §5),
                # pas la somme brute des points de cartes.
                team_points = {0: 0, 1: 0}
                team_points[capot_team] = 250
            else:
                team_points[team] += sum(card_point(c, trump) for c in cards)
            if belote_seat is not None and not belote_awarded:
                for p in trick['plays']:
                    if p['seat'] == belote_seat and p['card'][-1] == trump and p['card'][:-1] in ('K', 'Q'):
                        belote_played += 1
                if belote_played >= 2:
                    team_points[belote_seat % 2] += 20
                    belote_awarded = True
                    belote_awarded_trick = idx
            if capot_team is None and is_last and trump != 'TA':
                team_points[team] += 10
        running.append((team_points[0], team_points[1]))
    belote_info = None
    if belote_awarded_trick is not None:
        belote_info = {'team': belote_seat % 2, 'trick': belote_awarded_trick}
    return running, belote_info


def build_stages(history):
    stages = []
    hands = [list(h) for h in history['deal_hands']]
    initials = [list(h) for h in hands]
    for idx, trick in enumerate(history['tricks'], start=1):
        plays = [(p['seat'], p['card']) for p in trick['plays']]
        stage = {
            'trick': idx,
            'plays': plays,
            'winner': trick.get('winner'),
            'hands': [list(h) for h in hands],
            'completed': [history['tricks'][j]['plays'] for j in range(idx-1)]
        }
        for seat, card in plays:
            if card in hands[seat]:
                hands[seat].remove(card)
        stages.append(stage)
    return initials, stages


def player_box_html(seat, css_class, hand_cards, trump, leader_seat=None):
    label = PLAYER_POSITIONS[seat]
    is_leader = leader_seat is not None and seat == leader_seat
    extra_class = ' leader' if is_leader else ''
    badge = ' <span class="leader-badge" title="Entame du pli">&#9654; entame</span>' if is_leader else ''
    return f'<div class="player {css_class}{extra_class}">{label}{badge}<br>{hand_html(sort_hand(hand_cards, trump))}</div>'


def generate_page(history, out_path):
    initials, stages = build_stages(history)
    trump = history['contract'].get('trump')
    scores, belote_info = running_team_scores(history, trump)
    title = 'Coinche History Viewer'
    tabs = ['Auction'] + [f'Trick {i}' for i in range(1, len(stages)+1)]

    auction_cells = []
    for step in history['auction']:
        offer = step['offer']
        if offer is None:
            text = 'Pass'
        else:
            if isinstance(offer, tuple) and len(offer) >= 2:
                if len(offer) >= 4 and offer[3]:
                    text = '250'
                else:
                    text = f'{offer[0]}{offer[1]}'
            else:
                text = str(offer)
        auction_cells.append(f'<div class="bid-cell"><strong>{PLAYER_POSITIONS.get(step["seat"], step["seat"])}</strong><br/>{text}</div>')

    auction_html = '<div class="auction-grid">' + ''.join(auction_cells) + '</div>'
    hands_layout = ''
    for seat in range(4):
        hands_layout += f'<div class="hand-card hand-{seat}"><h4>{PLAYER_POSITIONS[seat]}</h4>{hand_html(sort_hand(initials[seat], trump))}</div>'

    stage_panels = []
    for i, stage in enumerate(stages, start=1):
        leader_seat = stage['plays'][0][0] if stage['plays'] else None
        played_cards_html = ''
        played_cards_list = []
        for seat, card in stage['plays']:
            pos = PLAYER_POSITIONS[seat].lower()
            played_cards_list.append(f'<div class="played-card {pos}">{card_html(card)}</div>')
        played_cards_html = '\n            '.join(played_cards_list)

        winner_label = PLAYER_POSITIONS.get(stage['winner'], str(stage['winner'])) if stage['winner'] is not None else '?'
        completed_html = ''
        if stage['completed']:
            completed_html = '<div class="completed-tricks"><h4>Tricks won</h4>'
            for tidx, trick in enumerate(stage['completed'], start=1):
                plays = ' '.join(f'{PLAYER_POSITIONS.get(p["seat"], p["seat"])}:{card_html(p["card"])}' for p in trick)
                completed_html += f'<div class="completed-item"><strong>{tidx}</strong>: {plays}</div>'
            completed_html += '</div>'
        score0, score1 = scores[i - 1]
        belote_badge_0 = ''
        belote_badge_1 = ''
        if belote_info is not None and i >= belote_info['trick']:
            badge = ' <span class="belote-badge" title="Belote (Roi+Dame d\'atout)">&#9734; belot&eacute;</span>'
            if belote_info['team'] == 0:
                belote_badge_0 = badge
            else:
                belote_badge_1 = badge
        panel = f'''
        <div class="panel" id="panel-{i}">
          <div class="board">
            {player_box_html(0, 'north', stage['hands'][0], trump, leader_seat)}
            {player_box_html(1, 'west', stage['hands'][1], trump, leader_seat)}
            {player_box_html(3, 'east', stage['hands'][3], trump, leader_seat)}
            {player_box_html(2, 'south', stage['hands'][2], trump, leader_seat)}
            {played_cards_html}
          </div>
          <div class="panel-footer">
            <div><strong>Trick</strong> {stage['trick']}</div>
            <div><strong>Winner</strong> {winner_label}</div>
            <div class="score-cumul">
              <strong>Score cumulé</strong>
              <span class="score-team score-team-0">{TEAM_LABELS[0]}: {score0}{belote_badge_0}</span>
              <span class="score-team score-team-1">{TEAM_LABELS[1]}: {score1}{belote_badge_1}</span>
            </div>
          </div>
          {completed_html}
        </div>
        '''
        stage_panels.append(panel)

    tabs_html = ''.join(
        f'<button class="tab-button{' active' if idx == 0 else ''}" data-target="panel-{idx}">{label}</button>'
        for idx, label in enumerate(tabs)
    )
    html = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>__TITLE__</title>
<style>
body { font-family: Arial, sans-serif; margin: 0; background: #f7f7f7; color: #222; }
.container { width: min(1200px, 100%); margin: auto; padding: 16px; }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.tab-button { padding: 10px 14px; border: 1px solid #bbb; background: white; cursor: pointer; border-radius: 6px; }
.tab-button.active { background: #0055aa; color: white; border-color: #004499; }
.panel { display: none; background: white; padding: 16px; border-radius: 12px; border: 1px solid #ddd; }
.panel.active { display: block; }
.auction-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 16px; }
.bid-cell { padding: 12px; border: 1px solid #ccc; border-radius: 10px; background: #fafafa; min-height: 72px; }
.board { position: relative; min-height: 320px; margin-top: 24px; }
.player { position: absolute; width: 220px; padding: 12px; border: 1px solid #ccc; border-radius: 12px; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
.player.north { top: 0; left: 50%; transform: translateX(-50%); width: 280px; }
.player.south { bottom: 0; left: 50%; transform: translateX(-50%); width: 280px; }
.player.west { left: 0; top: 50%; transform: translateY(-50%); width: 200px; }
.player.east { right: 0; top: 50%; transform: translateY(-50%); width: 200px; }
.player.leader { border-color: #0055aa; border-width: 2px; box-shadow: 0 0 0 3px rgba(0,85,170,.18); }
.leader-badge { color: #0055aa; font-weight: bold; font-size: .85rem; }
.played-card { position: absolute; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; }
.played-card.north { top: 50%; left: 50%; transform: translate(-50%, -80px); }
.played-card.south { bottom: 50%; left: 50%; transform: translate(-50%, 80px); }
.played-card.west { left: 50%; top: 50%; transform: translate(-80px, -50%); }
.played-card.east { right: 50%; top: 50%; transform: translate(80px, -50%); }
.panel-footer { display: flex; flex-wrap: wrap; gap: 24px; margin-top: 16px; align-items: center; }
.score-cumul { display: flex; gap: 12px; align-items: center; }
.score-team { padding: 4px 10px; border-radius: 8px; background: #eef6ff; font-weight: bold; }
.score-team-1 { background: #fff0ee; }
.belote-badge { color: #b8860b; font-weight: bold; font-size: .85rem; margin-left: 4px; }
.completed-tricks { margin-top: 20px; padding: 12px; border-radius: 12px; background: #eef6ff; }
.completed-item { margin-bottom: 8px; }
.card { display: inline-block; margin: 0 4px 4px 0; padding: 6px 8px; border-radius: 8px; color: white; font-weight: bold; }
.empty { color: #666; }
</style>
</head>
<body>
<div class="container">
  <h1>__TITLE__</h1>
  <div><strong>Contract:</strong> __CONTRACT__ by __TAKER__</div>
  <div class="tabs">
    __TABS__
  </div>
  <div id="panel-0" class="panel active">
    <h2>Auction</h2>
    __AUCTION__
    <h3>Initial hands</h3>
    <div class="board">
      __HANDSLAYOUT__
    </div>
  </div>
  __STAGEPANELS__
</div>
<script>
const buttons = document.querySelectorAll('.tab-button');
buttons.forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.tab-button').forEach(function(b) { b.classList.remove('active'); });
    document.querySelectorAll('.panel').forEach(function(p) { p.classList.remove('active'); });
    btn.classList.add('active');
    var target = btn.dataset.target;
    document.getElementById(target).classList.add('active');
  });
});
</script>
</body>
</html>
"""
    if history['contract'].get('capot'):
        contract_text = '250'
    else:
        contract_text = f"{history['contract']['level']}{history['contract'].get('trump')}"
    html = html.replace('__TITLE__', title)
    html = html.replace('__CONTRACT__', contract_text)
    html = html.replace('__TAKER__', str(PLAYER_POSITIONS.get(history['contract'].get('taker'), history['contract'].get('taker'))))
    html = html.replace('__TABS__', tabs_html)
    html = html.replace('__AUCTION__', auction_html)
    html = html.replace('__HANDSLAYOUT__', hands_layout)
    html = html.replace('__STAGEPANELS__', ''.join(stage_panels))
    Path(out_path).write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Render a Coinche history JSON into an interactive HTML viewer.')
    parser.add_argument('--history', default='game_history_test.json', help='Input history JSON path.')
    parser.add_argument('--out', default='game_history_viewer.html', help='Output HTML path.')
    args = parser.parse_args()
    with open(args.history, 'r', encoding='utf-8') as f:
        history = json.load(f)
    generate_page(history, args.out)
    print('Rendered', args.history, 'to', args.out)


if __name__ == '__main__':
    main()
