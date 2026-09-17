"""
Render a game to an animated GIF so it can be watched inside Jupyter (no display needed).

    from training.render import render_episode
    path, stats = render_episode("training/configs/heuristic_v0.json", seed=0, max_sim_time=300)
    from IPython.display import Image; Image(filename=path)

Rendering is for inspection only; scores from here are never logged.
"""
import json
from pathlib import Path

import training.episode  # noqa: F401  (sets headless SDL env vars before pygame is imported)
import pygame
from PIL import Image

from agents import make_agent
from training.episode import run_episode

ROOT = Path(__file__).resolve().parents[1]


def render_episode(config_path, seed: int, max_sim_time: float = 300, frame_every: int = 10,
                   width: int = 640, fps: int = 20, out_path=None):
    """frame_every=10 steps -> one frame per simulated second."""
    config = json.loads(Path(config_path).read_text())
    out_path = Path(out_path or ROOT / "logs" / f"render_{config['name']}_seed{seed}.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pygame.font.init()
    font = pygame.font.SysFont(None, 28)
    frames = []
    screen = None

    def on_step(sim, state, step):
        nonlocal screen
        if step % frame_every:
            return
        if screen is None:
            screen = pygame.Surface((width, int(width * sim.env_height / sim.env_width)))
        sim.env.draw(screen)
        text = (f"seed {seed}  t={sim.env.time:6.1f}s  score={state['score']:7.2f}  "
                f"agents={state['num_agents']}  predators={len(sim.env.predators)}")
        screen.blit(font.render(text, True, (255, 255, 255), (0, 0, 0)), (8, 8))
        frames.append(Image.frombytes("RGB", screen.get_size(), pygame.image.tobytes(screen, "RGB")))

    stats = run_episode(make_agent(config), seed, max_sim_time=max_sim_time, on_step=on_step)
    if frames:
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0,
                       optimize=False)
    print(f"{len(frames)} frames -> {out_path} | score={stats['score']:.2f} sim_time={stats['sim_time']:.1f} "
          f"status={stats['status']}")
    return str(out_path), stats
