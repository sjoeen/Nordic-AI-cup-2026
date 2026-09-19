# Late search-budget experiment

Standalone preserved eat/rest v1 with a conditional late-game search-speed budget. Original behavior is unchanged until the public deterioration trigger activates. Reproduction, food pursuit, escape, retirement and initial dispersal retain their original rules.

Normal full-game survival on seeds 1–3: 1702.5, 1625.5, 2002.5 seconds. Original: 1887.8, 1618.9, 1831.2. Mean 1776.8 versus 1779.3. The seed 3 result exceeds 2000, but seed 1 regresses. This is a promising development candidate for further investigation, not a demonstrated general improvement.

The engine and horizon were unchanged; historical baseline scores were reused. Public stock score/population checks match available pre-activation records on all three seeds. Death and meal diagnostics are observation proxies; repeated development seeds and engine tie-breaking variation limit conclusions.

See mechanism.md and protocol.json for precise trigger and behavior; results/ contains full normal-game logs and activation records. The included survival_agent.py is the tested standalone submission. The original preserved archive is unchanged.
