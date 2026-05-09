"""
Gera GIFs do agente treinado atuando em cada grid.
Execute: python generate_gifs.py
"""
import os
import sys
import glob
import numpy as np
import gymnasium as gym
from PIL import Image

os.makedirs("plots", exist_ok=True)

from gymnasium_env.grid_world_cpp_v2 import GridWorldCPPEnvV2
from stable_baselines3 import PPO
from train_grid_world_cpp_v2 import AntiLoopPredictor

try:
    gym.register(id="gymnasium_env/GridWorldCPPV2-v0", entry_point=GridWorldCPPEnvV2)
except Exception:
    pass


def record_episode(model, dim, obs_q, max_steps, max_frames=200, seed=None):
    """Roda 1 episódio e retorna lista de frames (numpy arrays)."""
    env = gym.make("gymnasium_env/GridWorldCPPV2-v0",
                   size=dim, obs_quantity=obs_q,
                   max_steps=max_steps, render_mode="rgb_array")
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    predictor = AntiLoopPredictor(model, deterministic=False)
    predictor.reset()

    frames = [env.unwrapped._render_frame()]
    done = trunc = False
    steps = 0
    while not done and not trunc:
        action = predictor.predict(obs, env.unwrapped._agent_location)
        obs, r, done, trunc, info = env.step(action)
        frames.append(env.unwrapped._render_frame())
        steps += 1

    env.close()

    # Subsample se passar do limite
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]

    return frames, info["coverage"], steps, (done and not trunc)


def save_gif(frames, path, duration_ms=120):
    """Salva lista de frames como GIF."""
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def main():
    models = sorted(glob.glob("data/ppo_*.zip"), key=os.path.getmtime)
    if not models:
        print("Nenhum modelo encontrado em data/.")
        sys.exit(1)

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = models[-1]

    print(f"Modelo: {model_path}\n")
    model = PPO.load(model_path, device="cpu")

    configs = [
        ("5x5",   5,  3,  200),
        ("10x10", 10, 12, 500),
        ("20x20", 20, 48, 2000),
    ]

    n_per_grid = 3

    for label, dim, obs_q, max_steps in configs:
        print(f"Gravando {label} ({n_per_grid} episódios)...")
        # Roda vários seeds e seleciona os n_per_grid melhores por cobertura
        results = []
        for seed in range(20):
            frames, cov, steps, full = record_episode(
                model, dim, obs_q, max_steps, max_frames=200, seed=seed)
            results.append((frames, cov, steps, full, seed))
            full_count = sum(1 for r in results if r[3])
            if full_count >= n_per_grid:
                break

        # Ordena: full coverage primeiro, depois por cov decrescente
        results.sort(key=lambda r: (-int(r[3]), -r[1]))
        chosen = results[:n_per_grid]

        for i, (frames, cov, steps, full, seed) in enumerate(chosen, start=1):
            out_path = f"plots/agent_{label}_ep{i}.gif"
            save_gif(frames, out_path, duration_ms=120)
            status = "FULL" if full else "parcial"
            print(f"  → {out_path}  (seed={seed}, {len(frames)} frames, "
                  f"cov={cov:.1%}, steps={steps}, {status})")

    print("\nGIFs salvos em plots/")


if __name__ == "__main__":
    main()
