# Survival controller candidates

Copy the files in this folder into your original survival-simulator directory, alongside `src/`. Keep the original simulator and its dependencies.

## Run the dispersal controller

```bash
python -m pip install -r requirements-agent.txt
python agent_dispersal.py
```

Then start your existing simulator in another terminal:

```bash
python simulation_server.py
```

The server uses the original interface: `POST http://127.0.0.1:9052/predict`.

## Try the alternatives

Stop the first agent server, then run:

```bash
python agent_decoy.py
```

Or try the simpler refuge controller:

```bash
python agent_refuge.py
```

All three controllers use the same port. Run one server process per simulation.

## Repeat the fixed-size local comparison

```bash
python benchmark_candidate.py --agent agent_dispersal
python benchmark_candidate.py --agent agent_decoy
python benchmark_candidate.py --agent agent_refuge
```

Each command runs seeds 1, 2 and 3 with the original default world, 0.1-second steps, and a 3,000-second limit. Each run stops at extinction or that limit. Survival time, score, survivors and completion status are printed to the terminal.

## What the candidates do

All use only the public agent statuses and observations. They combine persistent fruit allocation, obstacle-aware threat assessment, short trajectory comparisons for escape, controlled reproduction, and energy conservation.

The dispersal and decoy versions also spread newborn agents away from their birth location. The decoy alternative additionally uses some older agents to lead nearby hunters away from teammates, subject to its escape controller. Simulation resets and repeated identical HTTP requests are handled automatically.

None is a demonstrated full-simulation solution. The development comparison is small and extensively tuned on its three seeds, so additional seeds are needed to assess generalization. The original simulator also uses unordered object sets; exact outcomes can vary between processes even with a fixed seed.
