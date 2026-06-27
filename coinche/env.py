from typing import List, Optional
from .game import GameEngine
from .players import RandomPlayer

class CoincheEnv:
    """Minimal environment exposing history. reset() accepts optional hands (list of 4 lists of strings)."""

    def __init__(self, players=None, dealer: int = 0, agent_seat: int = 0):
        if players is None:
            players = [RandomPlayer(f"P{i}") for i in range(4)]
        self.players = players
        self.agent_seat = agent_seat
        self.engine = GameEngine(self.players, dealer)

    def reset(self, hands: Optional[List[List[str]]] = None):
        self.engine.deal(hands)
        self.engine.run_auction()
        obs = self._get_obs()
        return obs, {"history": self.engine.history}

    def _get_obs(self):
        agent = self.players[self.agent_seat]
        return {
            "hand": list(agent.hand),
            "trick": [],
        }

    def step(self, action: int):
        team_points, contract = self.engine.play()
        agent_team = self.agent_seat % 2
        reward = team_points[agent_team]
        done = True
        return self._get_obs(), reward, done, {"contract": contract, "history": self.engine.history}
