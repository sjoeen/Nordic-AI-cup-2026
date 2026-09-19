# neural-rl: GPU-trained variant of cluster-rl-training

Overlay for the `cluster-rl-training` package. `train_cluster_rl.py --learner neural` copies these
files to `<package>/neural/` inside a separate extraction (`~/neural-rl/cluster-rl-training`), so the
tabular package and its runs are untouched.

| File | Role |
|---|---|
| `policy_neural.py` | `NeuralRLPolicy(ForagingRLPolicy)`: 18 continuous features, MLP-ensemble Q values in pure Python, no in-game learning. Appended to the hash-checked baseline exactly like `hybrid.py`/`foraging.py`. |
| `train_gpu.py` | Fitted-Q trainer in PyTorch: all ensemble members batched with `baddbmm`, double-DQN targets, Huber loss, Adam. Falls back to CPU with a warning. |
| `neural_cluster.py` | CLI (`init`, `train`, `run`, `plan-eval`, `summary`, `export`, `worker`). Reuses the package's planner, worker, completion checks and summaries; replaces model validation, source building, the reducer and the code fingerprint (which now also covers these three files). |

Unchanged from the tabular experiment: stock simulator and HTTP `/predict`, full-length games, protected
modes, the four search-pace options, reward and credit rules, training from 600 s / evaluation from 900 s,
seed-split rules, immutable plans/checkpoints and resume behaviour.

Changed, and chosen by the executor rather than by a strategy spec (review before trusting results):

- State: 18 features instead of 7 bits. Network: 2 hidden layers x 32, ensemble of 5.
- Trainer: 20,000 steps/round, batch 4,096 per member, lr 1e-3, target sync every 500 steps, replaying
  the transitions of every round so far (the tabular reducer replayed only the newest round).
- No warm start: round 0 explores uniformly at random; an untrained network evaluates as pure baseline.
- Frozen-evaluation guard: leave the baseline only if mean ensemble advantage - 0.5 x ensemble std >= 0.1
  (replaces the visit-count guard; a heuristic, not a safety guarantee).

The GPU is used only by the trainer, for seconds to minutes per round. Games remain CPU-bound.
