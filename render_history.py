import argparse
import json
from pathlib import Path

PLAYER_POSITIONS = {0: 'North', 1: 'West', 2: 'South', 3: 'East'}

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


def hand_html(cards):
    if not cards:
        return '<span class="empty">(vide)</span>'
    return ' '.join(card_html(c) for c in cards)


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


def generate_page(history, out_path):
    initials, stages = build_stages(history)
    title = 'Coinche History Viewer'
    tabs = ['Auction'] + [f'Trick {i}' for i in range(1, len(stages)+1)]

    auction_cells = []
    for step in history['auction']:
        offer = step['offer']
        if offer is None:
            text = 'Pass'
        else:
            if isinstance(offer, tuple) and len(offer) >= 2:
                text = f'{offer[0]}{offer[1]}'
            else:
                text = str(offer)
        auction_cells.append(f'<div class="bid-cell"><strong>{PLAYER_POSITIONS.get(step["seat"], step["seat"])}</strong><br/>{text}</div>')

    auction_html = '<div class="auction-grid">' + ''.join(auction_cells) + '</div>'
    hands_layout = ''
    for seat in range(4):
        hands_layout += f'<div class="hand-card hand-{seat}"><h4>{PLAYER_POSITIONS[seat]}</h4>{hand_html(initials[seat])}</div>'

    stage_panels = []
    for i, stage in enumerate(stages, start=1):
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
        panel = f'''
        <div class="panel" id="panel-{i}">
          <div class="board">
            <div class="player north">{PLAYER_POSITIONS[0]}<br>{hand_html(stage['hands'][0])}</div>
            <div class="player west">{PLAYER_POSITIONS[1]}<br>{hand_html(stage['hands'][1])}</div>
            <div class="player east">{PLAYER_POSITIONS[3]}<br>{hand_html(stage['hands'][3])}</div>
            <div class="player south">{PLAYER_POSITIONS[2]}<br>{hand_html(stage['hands'][2])}</div>
            {played_cards_html}
          </div>
          <div class="panel-footer">
            <div><strong>Trick</strong> {stage['trick']}</div>
            <div><strong>Winner</strong> {winner_label}</div>
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
.played-card { position: absolute; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; }
.played-card.north { top: 50%; left: 50%; transform: translate(-50%, -80px); }
.played-card.south { bottom: 50%; left: 50%; transform: translate(-50%, 80px); }
.played-card.west { left: 50%; top: 50%; transform: translate(-80px, -50%); }
.played-card.east { right: 50%; top: 50%; transform: translate(80px, -50%); }
.panel-footer { display: flex; gap: 24px; margin-top: 16px; }
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
    html = html.replace('__TITLE__', title)
    html = html.replace('__CONTRACT__', f"{history['contract']['level']}{history['contract']['trump']}")
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
