# Medical Appointment Task Instructions

Work only inside this `medical-appointment` directory.

Do not inspect, modify, or run files in:
- `drone-flyby`
- `survival-simulator`
- the repository root
- deployment configuration
- systemd services
- the production server

Before changing code:
1. Read README.md.
2. Inspect api.py, dtos.py, example.py, utils.py, and local_evaluator.py.
3. Explain the proposed change.
4. Show the files that will change.
5. Wait for confirmation before making substantial changes.

Preserve the `/predict` route and the request/response schema unless explicitly asked to change them.

After changes:
- run the Medical Appointment local evaluator;
- run the oracle test when appropriate;
- run syntax/import checks;
- show the git diff;
- do not commit or push unless explicitly requested.

Never commit:
- `.venv/`
- `__pycache__/`
- model weights
- generated audio/transcription files
- secrets or API keys
