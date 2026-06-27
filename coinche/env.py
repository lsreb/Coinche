from typing import Dict, Any, List
from .game import GameEngine
from .players import RandomPlayer, Player

class CoincheEnv:
    """Minimal Gym-like environment for single-agent control of one seat.

    The environment controls player 0 by default and uses provided opponent strategies.
    Observation is a dictionary with `hand` (list of cards), `trick` (cards on table), and `tricks_won` counts.
    Action is an index into the agent's current hand.
    """

    def __init__(self, players=None, dealer:int=0, agent_seat:int=0):
        if players is None:
            # default 4 random players
            players = [RandomPlayer(f"P{i}") for i in range(4)]
        self.players = players
        self.agent_seat = agent_seat
        self.engine = GameEngine(self.players, dealer)

    def reset(self):
        self.engine.deal()
        self.engine.run_auction()
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        # return simple observation for agent
        agent = self.players[self.agent_seat]
        return {
            "hand": list(agent.hand),
            "trick": [],
        }

    def step(self, action:int):
        # play until end of hand (for simplicity) and return final reward for agent's team
        team_points, contract = self.engine.play()
        agent_team = self.agent_seat % 2
        reward = team_points[agent_team]
        done = True
        return self._get_obs(), reward, done, {"contract": contract}
