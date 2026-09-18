# Experimental overcrowding guard v4

New, separate experiment. All prior archives and source versions remain intact. The original rest/eat controller and its reproduction, movement, retirement, and food allocation are unchanged. Only pawn eligibility and mission cancellation differ from v3; lure movement is unchanged.

## Pawn activation

- At least 100 live agents (a provisional pawn activation threshold, not a population or birth cap).
- At least 30% of the whole population both below 40% of their own energy capacity and unmatched to distinct observed fruit within 160 units along an unobstructed route.
- Those conditions must persist for 3 seconds.
- The v3 candidate-quality, energy-runway, predator approach, and safe-route conditions must also pass. Retirement by itself is never enough.
- Existing missions stop if population falls below the threshold or the measured shortage clears.

The food calculation is an observation-only, greedy matching proxy. It cannot prove that all world food is exhausted or estimate long-term tree yield. Own fruit within eating distance is credited conservatively. The thresholds are provisional and have not been tuned for survival.

Each new selection records exact time, population, hungry-agent count, unmatched hungry count, and shortage duration in pawn_selection_events. The existing batch runner does not yet persist this additional event list automatically.

## Validation and remaining limitation

Nine decision checks passed: small starving populations (3/4/10/20/99) select zero pawns; a fed population of 100 selects zero; enough distinct nearby fruit blocks selection; a hungry population of 100 with no fruit activates a qualified pawn only after confirmation; shared food is not counted as many meals; population or food recovery cancels the role; no predator means no pawn. Ordinary actions, births, retirement sets, and configuration matched the baseline across 1,230 decisions.

These are constructed policy-state checks, not survival simulations. No scores or high-population survival improvements are claimed. Existing baseline results were reused. The unchanged original reproduction quota keeps the recorded populations far below 100. A meaningful natural large-population survival experiment therefore requires an explicit separate reproduction-policy decision; this package does not silently remove the quota.

Install requirements-agent.txt; run `python survival_agent.py` for the experimental server or `python locked_eat_rest_baseline.py` for the original. Do not run both on port 9052 together. Run `python -m unittest -v test_overcrowding_lure` for decision checks.

Do not delete or replace this or any prior output without explicit user permission.
