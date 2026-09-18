# Preserved original rest/eat controller

This is a separate, newly packaged copy of the unchanged original `loose_appetite250` controller. It contains no pawn overlay. The runnable source is byte-identical to the frozen original controller (SHA256 in provenance.json). Its original source snapshot is also included.

Install requirements-agent.txt, then run `python survival_agent.py` (port 9052, same /predict interface). The compatibility defaults for sim_time and n_agents are present.

Historical survival on seeds 1/2/3: 1887.8 / 1618.9 / 1831.2 seconds, using the original world size and 3000-second horizon. No simulations were rerun to create this archive.

This original version includes a reproduction quota based on a young-agent target starting at 6 and decaying toward 2, plus age retirement. That is not a hard ceiling on total population. These rules are preserved for reproducibility, not endorsed as the user's desired future policy.

Preservation instruction: do not remove, overwrite, or replace this archive or any earlier output without explicit user permission. New experiments must have separate names.
