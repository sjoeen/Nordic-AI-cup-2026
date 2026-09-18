import json
from pathlib import Path

from fastapi import FastAPI, Body

from src.utils.DTOs import StepResponse
from agents.heuristic_v1 import HeuristicAgentV1

HOST = "0.0.0.0"
PORT = 9052

SIM_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = SIM_ROOT / "training/configs/heuristic_v1.json"

app = FastAPI(title="Survival Simulator Agent Endpoint")

config = json.loads(CONFIG_PATH.read_text())
agent = HeuristicAgentV1(**config["params"])

@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    agent_states = [a.dict() for a in step.agent_status]
    actions = agent.act_batch(agent_states)

    # Must return {"actions": [...]} format
    return {"actions": actions}

@app.get("/")
def index():
    return {"message": "Agent endpoint running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)