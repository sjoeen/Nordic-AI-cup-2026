# C1-MECH-P02 — Deliberate facing offsets, pursuit, and predator sleep

(Verbatim task specification supplied by the human operator on 2026-09-17 13:48 +02:00.)

## Choose these Claude agents before starting

**Main executor: Claude Sonnet. Independent reviewer: Claude Opus.**

Start the implementation session on an available Sonnet tier in the existing authorized Claude account. Give it this entire file. Have it launch one read-only Opus reviewer with the reviewer brief below, explicitly selecting Opus. No Haiku worker is needed: this task has two substantive roles, and more agents would add coordination without a clear benefit.

If the interface cannot select an Opus subagent, run the reviewer brief in a separate Opus session and return its review to Sonnet. Do not assume an instruction inside a prompt changes the session's actual model. Record the actual model when exposed, otherwise mark it unverified. Preparation can proceed while review is pending; the main simulation matrix waits for the review. No new paid services or external API loops are authorized.

The reviewer inspects the design/source while Sonnet prepares the harness, then reviews the exact implemented revision before main execution. Sonnet alone edits implementation files. After results exist, the same reviewer checks interpretation. These reviews and fixes are part of the capped task, not an invitation to launch further experiments.

## Current objective and context

Read applicable repository instructions and `CLAUDE.md`. This prompt authorizes the next task after C1-MECH-P01, superseding that file's obsolete statement that P01 is the next task. Retain its reproducibility, baseline-protection, and evidence requirements.

We are learning predator behaviour before designing a population policy. The human's working idea is to lure predators away from the main population, potentially toward a corner. Sacrificing a decoy is acceptable if it buys useful protection. Individual survival is therefore one outcome, not the sole success criterion. We have not established that corners retain predators, or that relocation improves population survival.

C1-MECH-P01 reported:

- Collinear, exactly toward-facing agents usually triggered `circle` with `pivot_sign = -np.sign(rel_dir) = 0`, so the predator moved straight toward them. 1908/1936 circle ticks were this zero-angle case. Different branch labels did not mean different movement.
- In three matched pairs, off-axis reacquisition produced a real 45-degree pivot, perception loss, and greater separation. This is preliminary, not a confirmed strategy.
- Predators slow from 15 to 11 units/step below 40 energy. Their pursuit can consequently last much longer than an estimate based on continuous sprinting. Sleep in P01 occurred around 9.6–9.7 seconds in the cases where it occurred; no wake was observed within the 10-second horizon.
- All default walking arms were captured in that collinear setup. Higher-energy sprinting agents often survived by breaking contact. Escape and sustained transport are different objectives.
- Seven fresh-process and seven uninstrumented checks matched in the representative P01 fixture. This does not automatically validate a modified harness.

Additional source-inspection information supplied by the operator, to verify in this checkout:

- `src/elements/predator.py`, `Predator.step`, reportedly lines 16–100: nearest perceived target each step, without lock-on; chase if distance <90 or target facing angle >90 degrees; otherwise circle. Circle direction is target bearing plus a signed 45-degree pivot. Exact zero relative facing produces zero pivot.
- Perception: agents and walls only; hearing within 60 through walls; vision within 250 in a ±30-degree cone. Chase steering is limited, so it is not omnidirectional instantaneous pursuit.
- With no perceived agent but a visible wall, the predator steers away from that wall at walking speed 11. With neither, it moves forward at 11 and turns randomly by a small amount. Loss of contact does NOT immobilize a predator, and walls do NOT establish a permanent pen.
- `environment.py`, `non_agent_step`, reportedly lines 675–728: native energy costs and the <40 sprint clamp; sleep/recharge +3 per step; wake above 100 and act that tick; post-move kill distance <15; all agents in range can die; kills replenish energy up to 200. Sleep transition occurs after the kill check. Predators never die.
- Agent observations are generated before that tick's predator move, so the next controller action uses stale predator geometry. Default agent vision is 200 with a ±30-degree cone, hearing 50, max energy 500, walk/sprint 10/20, sprint disabled below 100. Movement is applied before turning.

These are local-starter claims. Equivalence to the official evaluator remains unverified. Verify source ordering and signs, and cite actual references; do not treat earlier line numbers as authoritative after code changes.

## Execution contract

[TYPE] RUN, including the limited diagnostic harness changes needed for this study.
[EXPERIMENT] C1-MECH-P02
[HYPOTHESIS] Maintaining a deliberate nonzero facing offset while retreating will exercise the actual circle pivot and may alter approach geometry or break predator perception at lower movement cost than direct escape. It may also lose control of the predator's travel direction or cause stale-observation oscillation. Survival and useful diversion are separate hypotheses.
[EXPECTED] Nonzero pivot ticks occur reproducibly in eligible states, unlike the zero-offset control. Matched traces reveal whether this changes capture, agent energy cost, continued pursuit, or predator displacement. A favourable outcome is not assumed.
[CONTROL] Exact-facing retreat (offset 0 degrees), at requested movement distances 10, 15, 20 per native step, using the fixed initial escape line and observation-based controller from P01. Measure these controls under P02's longer horizon.
[TEST] The same controller and fixtures with fixed signed facing offsets +15 degrees and -15 degrees. Offset magnitude is a predeclared diagnostic choice, not a parameter search; the signs test handedness and geometry. No adaptive offset or speed changes.
[SEEDS] 0,1,2 for synthetic development fixtures; no validation, final-holdout, or official-evaluation access.
[MODEL] Sonnet executor; Opus independent reviewer, for the roles specified above. Use available tiers, not invented version names.
[SUBAGENT] One Opus reviewer, read-only source/harness/result inspection. No runs or code edits by the reviewer; maximum two active Claude agents total.
[BUDGET] 90 minutes wall-clock total including both agents' work, review, implementation, execution, and reporting; concurrent work does not extend this deadline. Begin final reporting by minute 75, with no new episode launches after then. At most 10 minutes cumulative simulation wall-clock, one CPU simulation worker, at most 2 GB additional simulation memory, zero GPU use. Maximum 240 full episodes plus 12 smoke episodes of at most 5 native steps each. A full P02 episode has at most 400 native steps / 40 simulated seconds as defined below; the six P01 compatibility episodes have at most 100 steps / 10 seconds. Use existing dependencies and the existing Claude account; no downloads, new services, paid infrastructure, or external API loops. No open-ended retries or parameter search. Save partial results and stop at the first applicable cap.
[ACCEPTANCE] Complete the predeclared study with passing validation, or explicitly identify blocked, invalid, or missing cases. Apply the interpretation rules below. This accepts a measurement package, not a production policy. No score-improvement threshold or ship decision is requested.
[FILES] Inspect P01's `report.md`, `config.json`, `manifest.json`, existing traces/helpers, `training/predator_mechanics.py`, relevant simulator source, and applicable rules. Preserve P01 outputs and existing policies. Add a P02 harness module or backward-compatible extension and write task output under `experiments/C1-MECH-P02/`.

## 1. Source recovery and controlled arena

Use the current existing checkout; do not automatically switch branches or discard dirty changes. P01 ran on branch `main`, source revision `d9b5d9c`, harness `7b6805c`, Windows 11, Python 3.13.7 global, NumPy 2.2.5. Earlier handovers named another branch. Verify actual state and record differences. Use the same existing environment as P01 for this comparison if available; record its resolved versions rather than claiming it is pinned. Do not mix platforms/versions silently. Linux transfer is a separate later question, not an extra run matrix here.

Check existing official/local documentation for applicable local-testing constraints. Do not use official endpoints. If a required rule verification is unresolved, identify the specific blocked operation and continue permissible static preparation.

Retain native environment stepping, perception, movement, energy, sleep, kill order, and RNG behaviour. One default agent and one awake default predator; uniform grassland, no food, births, other actors, or new spawning. No production engine patches or rewritten predator physics. Keep P01's documented fixture-only spawn suppression and its RNG-gate behaviour.

Initial relative positions: agent (0,0), predator (d,0). Agent heads +x; predator heads -x. Agent age 0, max_age 120, default traits. Initial time 0, with a legitimate non-advancing initial observation. Initial escape direction is -x.

The longer horizon needs more clearance than P01. Use a 16000-by-16000 empty arena centered on internal (8000,8000), leaving boundaries outside possible travel and perception throughout 40 seconds. P01 used a larger chunk size to avoid an empty-edge visibility crash. If retaining that workaround, use chunk_size=8000 so far boundary edges remain candidates, and VERIFY that no edge enters either actor's observation or changes an action during these fixtures. Record this diagnostic deviation. Do not fix production `sensing.py` as part of this task.

Before main execution, validate this geometry/workaround against P01 using the compatibility checks below. If native behaviour cannot be preserved, stop the affected runs and report the blocker.

## 2. Exact controller and main matrix

Cross:

| Variable | Fixed values |
|---|---|
| Initial separation | 80, 120 |
| Agent energy | 150, 300 |
| Predator energy | 101, 200 |
| Requested movement distance each step | 10, 15, 20 |
| Facing offset delta | 0, +pi/12, -pi/12 (0, +15, -15 degrees) |
| Seed | 0,1,2 |

8 initial-state fixtures × 9 controller arms × 3 seeds = **216 main episodes**. Prioritize the measured zero-offset controls, then execute the fixed signed-offset cases. Save the run order in the configuration before execution. Do not add more angles, speeds, reserves, or cases based on early results.

At every controller action:

1. Request movement along the FIXED initial escape line: `move_direction = wrap(pi - heading_accumulator)` and `move_distance = arm_speed`. Track heading from issued turns, initialized to zero. No world truth is needed to preserve that line.
2. If the single predator is in the actual agent observation with local bearing `beta`, request `turn_angle = wrap(beta + delta)`. This makes the commanded heading offset by delta from the observed predator bearing. If it is absent, request turn=0. Do not scan, predict its hidden position, or move toward a last known position.
3. Update the heading accumulator after issuing the turn. Use canonical wrapping [-pi,pi). Preserve native movement-before-turn ordering and the stale observation; do not recompute an up-to-date bearing from engine truth.
4. No spawning; exactly one action per living agent each native step. Keep speed fixed even when the engine clamps it. No scripted reaction to predator energy, sleep, target state, or hidden location.

Delta is a desired offset from an OLD observed bearing; it is NOT guaranteed to equal the true relative facing angle when the predator decides. Log both. Verify circle side from actual source conventions, not from an assumed positive/negative interpretation. The same initial distance also need not be the distance at the predator's decision after agent movement.

Run the living-agent phase until death or 30 seconds / 300 native steps, whichever comes first. Continue through sleep, wake, perception loss, and reacquisition. An event absent by the horizon is censored. Do not guarantee a full cycle if the predator has not slept or the agent dies first.

### Post-capture observation, without new actors

If the agent dies before 30 seconds, continue the native predator/environment dynamics for **10 additional seconds / 100 native steps**, with no agent actions or replacement target. Log the predator's behaviour after the decoy is gone. Use the driver to continue ordinary environment steps, not a rewrite of predator behaviour or a patch to the official termination rules.

This continuation is explicitly COUNTERFACTUAL relative to the one-agent game's extinction termination. It is a local behavioural probe of an empty area after a decoy disappears; it is not extra species survival or a simulation of a living protected group. Label its time axis relative to death and keep it out of survival metrics. If continuing the real native steps is not possible without changing dynamics, omit this phase and report why rather than inventing behaviour.

Do not remove a surviving agent at 30 seconds, force a kill, or force a loss of perception. For survivors, use naturally occurring loss-of-contact periods within the 30-second trace. Record the available follow-up length; do not treat them as a matched forced-release experiment.

## 3. Independent review and bounded validation

Allow at most 12 smoke episodes of at most 5 ticks for source/controller/measurement checks. The reviewer must check the IMPLEMENTED revision before the main matrix. Routine implementation fixes that restore this specification are authorized; changes to its hypothesis, matrix, controls, or budget are not.

Required checks include:

- At d=120 with ample energy, both signed-offset controllers induce a nonzero pivot at an eligible decision, with correctly mirrored initial circle sides. Do not demand nonzero pivot on every later tick, where geometry/staleness can change it.
- Turn, local movement conversion, desired/true facing, native perception cones, and move-before-turn timing are correct.
- At decision distance <90, the source's chase rule still holds regardless of offset.
- Below-40 predator clamping, kill-before-sleep ordering, and wake-and-act timing are logged correctly. Distinguish predator energy just BEFORE a kill from energy AFTER eating.
- The logger never mutates state or consumes RNG. No boundary edge enters observations.

Additional full episodes, all already budgeted:

**Six P01 geometry compatibility runs:** in the ORIGINAL P01 arena and controller, run zero-offset speeds 10,15,20 for d120/aE150/pE200 and d120/aE300/pE101, seed0, maximum 10 seconds each. Compare with the first 10 seconds or common pre-death interval of the matching P02 zero-offset runs. Compare normalized physical states, actions, observations, decisions, energies, event ticks, and RNG state. Absolute coordinates/grid bookkeeping will differ by construction. Require identical discrete actions/modes/events and numerical agreement within absolute 1e-8 for physical quantities; report actual maximum errors. Retain original P01 data, and also compare existing historical traces where available. Do not demand identical whole-world hashes across different arena layouts.

**Nine fresh-process repeats:** representative P02 fixture d120/aE300/pE101, seed0, all 9 controller arms. Compare exact trajectory digests against main runs on the same fixture geometry, excluding wall-clock/log metadata.

**Nine instrumentation-disabled repeats:** the same 9 cases, using minimal state/action/RNG digests instead of detailed telemetry. Require equality with the instrumented runs.

Total full episodes: **216 + 6 + 9 + 9 = 240**. Perform early validation on the representative subset and relevant controls before completing the remaining matrix; each main case counts only once. If a compatibility, repeat, or instrumentation check fails, preserve evidence, stop remaining matrix execution, and report. Do not consume extra unbudgeted runs to chase a result.

## 4. Required measurements

Log controller observations/actions separately from privileged diagnostic truth. Assign one diagnostic predator identifier only in logs. Do not expose new signals to the controller.

Per native tick, record phase-aligned positions/headings, agent and predator energy, commanded/realized movement, clamp status, desired facing offset, true relative facing at predator decision, predator target/perception status, chase/circle/no-target mode, pivot sign, sleep/wake, and pre-kill and post-kill distances/energies. Audit the emitted decision against the exact source predicate. If mode is reconstructed rather than directly observed, label that and validate its provenance; do not claim branch instrumentation from a post-step picture alone.

Summarize separately:

**Mechanism:** eligible circle ticks, zero versus nonzero pivot ticks, pivot sign changes, actual relative-angle distribution, agent perception loss, predator perception loss/reacquisition, and time in each mode. A changed label without changed motion is not evidence of a useful effect.

**Agent outcome/cost:** capture or energy death and time; survival censored at 30 seconds; minimum separation; energy used and turning cost; sprint-clamp onset; energy/separation at first sleep, first wake, and horizon/death. Separate missing events from numeric zero. At first wake state whether the agent was still alive, perceived, and sprint-capable, and its outcome for the remaining observed duration.

**Predator transport:** use the initial agent position as a fixed reference point and -x as the requested transport direction. Record predator displacement projected along -x, distance from that reference, lateral displacement, and actual predator-targeted time. Distinguish predator movement while targeting the decoy from movement while targetless. Record these at first perception loss, capture/death, sleep, wake, and horizon where present. Large agent-predator separation alone is NOT transport success.

**After contact ends:** for natural perception-loss intervals record how long the predator remains targetless, how far it travels, whether it reacquires, and its closest subsequent approach to the initial reference point over the available interval. For the post-death phase record position/displacement/energy immediately before and after the kill (if captured), then at +1,+5,+10 seconds; sleeping/waking; and minimum distance to the initial reference point. The reference is not an actual safe group or resource patch, and must not appear as a perceived actor.

Report paired delta+15 minus delta0 and delta-15 minus delta0 within the same fixture, speed, and seed. Compare energy/displacement at common elapsed times while both agents are alive; present different death times separately. Also compare the two offset signs as a handedness diagnostic. Do not assume exact full-trajectory mirroring after random wandering unless RNG transformations justify it.

## 5. Interpretation, statistics, and failure conditions

This is exploratory mechanism evidence, not a search for an optimal angle or proof of improved official score.

- First ask whether the intended nonzero circle motion was actually exercised. If not, explain the controller/geometry/observation reason; do not expand the angle grid.
- Then ask whether it improves any useful dimension: survival, energy cost, sustained pursuit toward the chosen direction, or time spent away after loss/capture. Report tradeoffs rather than inventing a combined reward.
- Sacrificial cases can show useful displacement, but without another agent or feeding patch there is no measured protection benefit. A caught decoy is neither automatically a failure nor automatically a success.
- A sleeping predator is not permanently neutralized. Evaluate observed wake behaviour and distinguish it from a timer estimate.
- Predators wander at speed 11 without a target and steer away from visible walls. This open-arena study cannot prove corner containment; do not add corners or wall tests to this matrix.
- Report n, mean, sample std, median, min, max for appropriate per-fixture continuous metrics, matched deltas, event counts, and censoring. Do not average censored death times as if they were observed deaths, pool unlike fixtures into a headline win rate, or treat identical deterministic seeds as independent generalization samples.
- Keep source facts, measurements, calculations, and hypotheses distinct. Each strategic inference must state its evidence and what would overturn it. No production-policy promotion.

## 6. Opus reviewer brief — pass this with the full prompt

[TYPE] REVIEW
[EXPERIMENT] C1-MECH-P02
[MODEL] Opus — independent scrutiny of geometry, observation timing, measurement validity, and interpretation.
[SUBAGENT] NONE; do not delegate again.
[BUDGET] Within the parent task's 90-minute cap: at most 20 minutes on design/source/harness review and 10 minutes on result review; zero simulation runs, zero edits to implementation, no new services.

Read this specification and the exact relevant source/harness revision. Review independently rather than merely checking whether Sonnet says it complied. Focus on: offset sign and genuine circle motion; agent vision/stale observations; movement/turn order; arena enlargement/chunk workaround; energy phase around a kill; first sleep/wake; matched controls; contamination from privileged truth; and the meaning of post-extinction continuation.

Return a concise list of blocking validity defects and nonblocking limitations, with evidence and exact references, or state no blocking defect found. Do not redesign the strategy, change the matrix, or tune parameters. Sonnet fixes conformance defects and supplies the revised diff for your bounded recheck. Main execution begins only after the implemented revision has no unresolved blocking defect. If the cap or unavailable reviewer prevents that, report the prepared work and blocker instead of claiming review occurred.

After execution, inspect report/summary and representative traces within your remaining review budget. Check that conclusions concern observed motion, not merely branch labels; that pre-kill energy is not confused with replenished energy; and that escape, transport, sacrifice, and containment are not conflated. Give a short findings memo for inclusion in the final report. No additional runs.

## 7. Outputs and stopping point

Write under `experiments/C1-MECH-P02/`:

- Exact immutable configuration, execution order, and commands; manifest with source/harness commits and hashes, interpreter/dependencies, actual models, caps, usage, fixture deviations, and run validity.
- Per-tick traces, episode summaries, paired comparisons, compatibility/repeat results, and review findings. Preserve failures and unexecuted-case counts. Do not overwrite P01 or write synthetic scores into the competition results index.
- Legible distance/energy plots with sleep/wake markers and capture/threshold annotations; equal-aspect trajectory plots showing initial positions, escape direction, and separate post-death paths; and a pivot/perception timeline that exposes genuine circling versus straight zero-pivot motion. Use existing dependencies only. Do not rely on overlapping line styles to show the offset effect: include paired-difference plots or separate panels.
- `report.md`, answering: did offsets cause true circling; did walking ever remain viable through a wake; what did extra turning cost; when did pursuit become wandering; how far was the predator actually transported; what happened after a sacrifice; which conclusions survive the well-fed predator case; and what remains unknown about corners, multiple predators, protected agents, and official-evaluator equivalence?

Commit task-owned harness/config changes before the main matrix and the report/compact summaries afterward. Preserve unrelated dirty work and baseline/submission files; no pushing, merging, deployment, or policy changes. Include exact reproduction commands and trace paths for favourable and unfavourable cases if both exist. Do not start a follow-on experiment.

Final response: [STATUS], [EXPERIMENT], [ARTIFACT], [RESULT], [CONCLUSION], [FLAGS]. Name the models actually used, give completed/invalid/missing counts, and link the written report. Save a partial report even if execution is blocked or capped. Then stop for strategy discussion.
