"""Run beside the original simulator's src directory; same fixed-size benchmark."""
import json
import os
import statistics
import time

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')

from src.core import SimulationCore
import argparse
import importlib


def run(seed):
    simulation=SimulationCore(seed=seed)
    policy=make_policy()
    actions=[]
    started=time.perf_counter()
    for tick in range(30000):
        state=simulation.step(actions)
        if not state['num_agents']:break
        decoded=policy.decide_all(state['observations'],state['sim_time'])
        actions=[(action.agent_id,action) for action in decoded]
        if (tick+1)%2000==0:
            print(json.dumps(dict(event='progress',seed=seed,sim_time=round(state['sim_time'],1),
                                  alive=state['num_agents'])),flush=True)
    result=dict(seed=seed,survival_seconds=round(state['sim_time'],1),score=state['score'],
                completed=bool(state['num_agents'] and tick+1==30000),alive=state['num_agents'],
                wall_seconds=round(time.perf_counter()-started,2))
    print(json.dumps(dict(event='result',**result)),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--agent',choices=('agent_dispersal','agent_decoy','agent_refuge'),default='agent_dispersal')
    args=parser.parse_args()
    make_policy=importlib.import_module(args.agent).make_policy
    results=[run(seed) for seed in (1,2,3)]
    print(json.dumps(dict(event='summary',mean_survival_seconds=statistics.mean(r['survival_seconds'] for r in results),
                          completions=sum(r['completed'] for r in results),runs=len(results))),flush=True)
