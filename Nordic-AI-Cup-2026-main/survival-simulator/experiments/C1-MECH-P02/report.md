# C1-MECH-P02 -- Deliberate facing offsets, pursuit, and predator sleep

Source: `challenge-1` (identical to `main@d9b5d9c`). Harness commits: `702bf2c`
(prep), `261cb00` (2 Opus BLOCKING fixes), `7219796` (episode-budget fix), plus
one more commit alongside this writeup (fixed a false-positive compat failure
caused by this session's own comparison code). Interpreter/deps: CPython
3.13.7, numpy 2.2.5, scipy 1.16.2, shapely 2.1.2, pygame 2.6.1, pydantic
2.12.5, matplotlib 3.10.6. Config: `config.json` (predeclared matrix, run
order, caps). Seeds: 0, 1, 2. Reproduction: `python -m
training.predator_mechanics_p02 main --phase 1`, then `compat`, `repeat
--tag repeat`, `repeat --tag nolog`, then `main --phase 2`, then
`python experiments/C1-MECH-P02/analyze.py`.

Models: Sonnet executor (this session), Opus reviewer (separate
orchestrator-relayed session; stage-1 design review, stage-2 validity
recheck, a human hard-cap correction, and a pending final conclusions check).
Three Claude sessions were active for this experiment (executor, reviewer,
orchestrator).

## Validity checks (all passed -- see `manifest.json` for detail)

- Main matrix (216 episodes): 0 predicate/observed-mode mismatches, 0
  edge-avoid-eligible ticks, 0 edges ever perceived by either actor, across
  every tick of every episode.
- Compat (6 P01-arena episodes vs. the matching already-run P02 zero-offset
  main episodes): all 6 pass; max absolute error over all compared
  position/heading/energy/action/decision fields = 3.2e-12 (tolerance
  1e-8); RNG state identical at the truncation tick for all 6; outcomes
  consistent; also matches the historical P01 traces in the main checkout.
  This is strong, direct evidence that P02's 16000-arena/chunk_size=8000
  geometry reproduces P01's native engine behaviour exactly for the
  zero-offset controller.
- Repeat (9 fresh-process) vs. main; nolog (9, instrumentation-disabled)
  vs. repeat: 0 mismatches in both.
- One compat run initially reported a failure that turned out to be a bug in
  this session's own comparison code (`_compare_ticks` hardcoded which side
  was "P01-format" instead of taking it as a parameter, breaking the
  historical-P01-vs-fresh-P01 comparison specifically). Fixed and the
  6-episode compat matrix re-run to completion (deterministic, so the first
  5 of the 6 reproduce bit-for-bit); not a simulator or harness defect. Full
  detail in `manifest.json`.

## 1. Did offsets cause true circling?

Yes. At the eligible tick-1 decision (agent still perceived, predator not
yet within chase range), the signed offsets produce the derived,
counter-intuitive-but-verified sign relation: true rel_dir = -delta,
pivot_sign = sign(delta), and the predator's resulting displacement carries
sign -sign(delta) -- the predator circles to the side opposite the
agent's gaze offset, not the same side. This was verified exactly
(rel_dir = plus-or-minus 0.2617993877991496 rad = 15 degrees, to 1e-15) in
both the smoke checks and the full run, at both d=120 and the borderline
d=80 case.

Across all 216 episodes, mean nonzero-pivot ticks per episode was 51.0
for both +15 degree and -15 degree arms vs. 1.96 for the 0 degree control
(the control's rare nonzero pivots come from floating-point residue after
perception is briefly re-established off-axis, not the collinear-zero case).
pivot_sign never flipped sign within a single episode (0 sign changes over
7,665 circle-mode decision ticks) -- once established, the pivot side is
stable, consistent with the reviewer's predicted "reinforcing, not
self-cancelling" drift. The true relative-facing angle at decision spread
over plus-or-minus 63 degrees (std 20.3 degrees) once circling was
underway, confirming the predator's bearing genuinely moved, not just its
branch label.

## 2. What changed in survival and energy cost?

No episode in this matrix survived to the 30s horizon -- every one of the
216 episodes ended in either capture (117, 54 percent) or energy death (99,
46 percent). By offset: control (0 degrees): 45 captured / 27 energy-death
(of 72); each signed offset: 36/36 (of 72). So the offsets shift roughly a
fifth of would-be captures into energy deaths instead -- consistent with
increased separation (mean minimum separation 47.4 vs. 35.6 for the
control) bought at a real, if small, cost: mean total agent turn-cost per
episode 0.229 vs. 0.0 for the control (delta=0 keeps the commanded turn
near zero whenever the predator is dead ahead). At matched elapsed time
while both arms' agents were alive, the offset arms showed +35.5 units
more separation at t=1s and -0.106 more energy spent than the control,
growing to +71 to +103 units separation and -0.14 energy by t=5s (fewer
pairs both survive that long). In no matched pair among "both captured"
cases did an offset arm get captured sooner than its control (min observed
delta = +0.1s). The two signs are handedness-mirrored: aggregate outcome
counts, mean minimum separation, and mean turn cost are identical between
+15 and -15 degrees to the reported precision; at the trajectory level,
30/48 (62.5 percent) of matched-alive pairs at t=5s were exact mirrors, the
rest had diverged after independent perception-loss/wander RNG draws.

## 3. Did the predator keep following, or lose contact?

Mostly kept following. Across 216 episodes there were only 120
predator-side perception losses and 22 reacquisitions (vs. 117 agent-side
losses / 18 reacquisitions from the controller's perspective) -- under 0.6
losses per episode on average, and once lost, reacquisition happened well
under half the time within the remaining horizon. Total time-in-mode summed
over all episodes: chase 306s, circle 767s, wander 1,077s, sleep 278s -- a
substantial share of wander time, but that is the predator's wander
(no target), not evidence the agent escaped for long; most wander
intervals precede eventual energy death, not lasting freedom.

## 4. What happened through sleep/wake, and after capture?

82/216 episodes (38 percent) included a predator sleep event; 81 of those
(99 percent) also showed a wake within the living 30s phase -- a much
higher observed wake rate than P01's 10s horizon, simply because the
horizon is 3x longer. Example: fixture d=80/aE=150/pE=101, arm S15_D+15,
seed 0 -- predator first sleeps at 9.6s, wakes at 13.1s, and the agent
still dies (of its own energy) afterward, not of a second capture in this
trace. A sleeping predator is confirmed not permanently neutralized in
this longer horizon, as expected.

All 216 episodes ended before the 30s horizon, so all 216 got the
10s/100-step post-death continuation. This phase is native
step_environment(env, [], dt) calls with no agent and no controller
action -- purely a counterfactual probe, excluded from survival metrics.
Only 4 total wake events occurred during the 216 post-death phases combined
(most captured-outcome predators are near full energy right after eating
and simply don't sleep again; most energy-death-outcome predators' sleep
state carries over from the living phase). The predator's closest approach
to the fixed reference point (the agent's initial position) during the
post-death window averaged 1,048 units away (min 141, max 2,337) -- the
predator does not return toward the original location once the decoy is
gone; it continues wandering wherever the chase left it.

## 5. What does this support, or fail to support, about decoys?

Supports: a facing offset while retreating reliably produces a real,
non-label-only change in predator motion (genuine circling, not just a
renamed straight approach), and reliably buys modestly more separation
early in the chase, at a small, quantifiable energy cost, without ever
making capture happen sooner in the matched comparisons run. The predator,
once engaged, tends to stay engaged with whatever it was chasing rather
than reverting toward its start; after a kill or after the target
disappears, it does not autonomously drift back to the original area within
10s.

Does not support: any claim of protection -- there is no second agent
or resource patch in this matrix, so "the predator was pulled away" has no
one to protect. It also does not support that offset retreat improves
raw survival: 0/216 episodes survived to the horizon regardless of arm, and
the offsets increase the fraction of energy deaths relative to captures
rather than reducing overall lethality. A caught decoy fully replenishes the
predator (energy capped at 200) -- this matrix does not show whether that
replenishment matters for a second, protected agent, because there isn't
one here.

## Known limitations (disclosed, not fixed here)

- dist_post_step on an energy-death tick uses the removed agent's
  last-known (frozen) position rather than being nulled; this is the
  genuine last observed separation, not fabricated, but should not be read
  as a "kill distance" -- energy deaths have no kill.
- The post-death continuation (10s/episode) is not covered by the
  repeat/nolog fresh-process/instrumentation-disabled digest checks, which
  only exercise the living-phase-equivalent 100 steps.
- Smoke budget: 12/12 used, 0 in reserve -- 10 from the designed
  cmd_smoke run plus 2 ad hoc probes made while diagnosing and fixing one
  smoke case's own fixture-parameter bug (mid-run correction, both attempts
  kept in smoke.json, nothing hidden).
- The orchestrator relayed between three active Claude sessions (this
  executor, an Opus reviewer, and the orchestrating session) across two
  review passes plus one hard-cap correction; all fixes and reruns are
  recorded in the git history and manifest.json.
- Linux / official-evaluator equivalence remains unverified (out of scope,
  as in P01). This remains a synthetic one-predator/one-agent mechanism
  study; multiple predators, protected agents, and corner/wall containment
  are explicitly out of scope per the spec.

## Representative traces (for the reviewer)

All under experiments/C1-MECH-P02/traces/:

- Nonzero-pivot / mirrored-handedness case: d120_ae300_pe101_S10_D+15_s0.json.gz
  and ..._S10_D-15_s0.json.gz (tick 1: pivot_sign = +1.0 / -1.0,
  rel_dir = plus-or-minus 0.26180 rad exactly).
- Favourable-ish case (offset delayed death by 12.4s, though still to
  energy death, not survival): d80_ae150_pe101_S15_D+15_s0.json.gz vs.
  ..._S15_D0_s0.json.gz.
- Unfavourable/neutral case (offset bought large late separation but no
  change in death timing): d80_ae300_pe101_S15_D+15_s0.json.gz vs.
  ..._S15_D0_s0.json.gz.
- Sleep/wake case: d80_ae150_pe101_S15_D+15_s0.json.gz (sleep at 9.6s,
  wake at 13.1s, agent dies of energy afterward -- same file as the
  favourable case above).
- Post-death case: any file has a post_death_ticks list; the plotted
  representative is d120_ae300_pe101_S10_D{0,+15,-15}_s0.json.gz (see
  plot_trajectories.png).

Figures (max 3, per the efficiency amendment): plot_timeline.png
(separation + pivot-sign timeline, representative fixture), plot_trajectories.png
(equal-aspect trajectories including post-death paths), plot_paired_difference.png
(paired delta-vs-control separation effect at t=5s). Full outputs:
summary.csv, stats_by_fixture_arm.csv, matched_comparisons.csv,
analysis_census.json, repeat_nolog_compat_report.json, compat.json,
digests_main/repeat/nolog.json, manifest.json.
