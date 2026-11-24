from gga.protocol import Action, Observation, ToolSpec

from gaming_agent.policy import Policy


class Agent:
    def __init__(self):
        self.policy = Policy()

    def step(self, obs: Observation, tools: list[ToolSpec]) -> Action:
        tool, params = self.policy.step(obs, tools)
        return Action(tool=tool, params=params)
