1. python play_test.py --strategies random,random,random,random --out game_test.json 
crée le fichier test

2. python3 render_history.py --history game_test.json --out game_history_viewer.html 
génère une représentation graphique à partir d'un historique

3. python -m coinche.players 
par exemple, pour faire des tests sur un fichier