import sys
from pathlib import Path

from fastapi import FastAPI, Body

from src.utils.DTOs import StepResponse

HOST = "0.0.0.0"
PORT = 9052

SIM_ROOT = Path(__file__).resolve().parent
CANDIDATE_DIR = SIM_ROOT / "external/candidates/original-eat-rest-overcrowding-v4"
if str(CANDIDATE_DIR) not in sys.path:
    sys.path.insert(0, str(CANDIDATE_DIR))

# same adapter pattern as external/candidates/run_candidate.py's CandidateAdapter
import survival_agent  # noqa: E402  (eat-rest-overcrowding-v4)

app = FastAPI(title="Survival Simulator Agent Endpoint")

agent = survival_agent.make_policy()

@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    agent_states = [a.dict() for a in step.agent_status]
    actions = agent.decide_all(agent_states, step.sim_time)

    # Must return {"actions": [...]} format
    return {"actions": [a.model_dump() for a in actions]}

@app.get("/")
def index():
    return {"message": "Agent endpoint running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)