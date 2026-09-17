"""
Agent policies. Every agent exposes the same interface as the submission endpoint:

    agent.act_batch(agent_states: list[dict]) -> list[dict]

where each state is an ObservationResponse dict (as sent by the evaluator) and each
returned dict has the ActionRequest fields.
"""
import json
from pathlib import Path

from agents.dummy import DummyAgent
from agents.heuristic import HeuristicAgent

AGENTS = {"dummy": DummyAgent, "heuristic": HeuristicAgent}


def load_config(path) -> dict:
    with open(path) as f:
        return json.load(f)


def make_agent(config: dict):
    return AGENTS[config["agent"]](**config.get("params", {}))
