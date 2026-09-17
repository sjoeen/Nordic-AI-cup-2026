"""
Render a game to an MP4 (GIF fallback) so it can be watched inside Jupyter (no display needed).

    from training.render import render_episode
    path, stats = render_episode("training/configs/heuristic_v0.json", seed=0, max_sim_time=300)
    from IPython.display import Video; Video(path, embed=False)   # linked, keeps the notebook small

MP4 needs `imageio` + `imageio-ffmpeg` (bundled ffmpeg binary); without them a GIF is written,
which is much larger because the biome background is per-pixel random colour.

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
    """frame_every=10 steps -> one frame per simulated second. width should be a multiple of 16 (4:3 map)."""
    config = json.loads(Path(config_path).read_text())
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
        ext = "mp4"
    except ImportError:
        imageio, ext = None, "gif"
    out_path = Path(out_path or ROOT / "logs" / f"render_{config['name']}_seed{seed}.{ext}")
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
    if frames and out_path.suffix == ".mp4":
        import numpy as np
        with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7, macro_block_size=16) as w:
            for fr in frames:
                w.append_data(np.asarray(fr))
    elif frames:
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"{len(frames)} frames -> {out_path} | score={stats['score']:.2f} sim_time={stats['sim_time']:.1f} "
          f"status={stats['status']}")
    return str(out_path), stats
