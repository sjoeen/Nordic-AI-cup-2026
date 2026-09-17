import random

from src.utils.controllers.dummy_agent_policy import action_decision


class DummyAgent:
    """Starter dummy policy, reproduced exactly as the starter agent_server.py serves it
    (a fresh Random(1) per request, shared by all agents in that request)."""

    def act_batch(self, agent_states):
        rng = random.Random(1)
        return [action_decision(s, rng).model_dump() for s in agent_states]
