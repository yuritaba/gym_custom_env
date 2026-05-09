#
# python train_grid_world_cpp_v2.py <modo> [dim]
#
# Modos disponíveis:
#   train        — curriculum completo do zero (stages 1 → 6)
#   continue     — continua do melhor modelo salvo (stages 7 → 9: 10×10 + 20×20)
#   test   [dim] — avalia modelo salvo em 100 episódios no grid DIMxDIM
#   run    [dim] — roda um episódio com visualização
#
# Exemplos:
#   python train_grid_world_cpp_v2.py train
#   python train_grid_world_cpp_v2.py continue
#   python train_grid_world_cpp_v2.py test 5
#   python train_grid_world_cpp_v2.py test 10
#   python train_grid_world_cpp_v2.py test 20
#   python train_grid_world_cpp_v2.py run 10
#

import os
import sys
import glob
import gymnasium as gym
import numpy as np
from collections import deque
from datetime import datetime

os.makedirs("data", exist_ok=True)
os.makedirs("log",  exist_ok=True)

from gymnasium_env.grid_world_cpp_v2 import GridWorldCPPEnvV2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

# ─── Registro do ambiente ─────────────────────────────────────────────────────

def _register():
    try:
        gym.register(id="gymnasium_env/GridWorldCPPV2-v0", entry_point=GridWorldCPPEnvV2)
    except Exception:
        pass

_register()

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make(dim, obs, max_steps):
    def _f():
        return gym.make("gymnasium_env/GridWorldCPPV2-v0",
                        size=dim, obs_quantity=obs,
                        max_steps=max_steps, render_mode="rgb_array")
    return _f

def _make_vec(env_fns):
    return DummyVecEnv(env_fns)

def _print_action(a):
    return {0:"right", 1:"up", 2:"left", 3:"down"}.get(a, "?")

# ─── Anti-loop predictor (opção 1+2: stochastic + loop break) ────────────────

_ACTION_DIRS = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}
_LOOP_WINDOW = 8


class AntiLoopPredictor:
    """
    Wrapper de inferência que combina:
    1. Sampling estocástico (deterministic=False) — quebra empates entre frontiers
    2. Detector de loop de 2 células — força ação não-back quando preso

    Mantém histórico das últimas N posições. Se em N passos o agente visitou
    apenas <= 2 células únicas, considera-se em loop e amostra uma ação que
    NÃO retorne à posição imediatamente anterior.
    """
    def __init__(self, model, deterministic=False, window=_LOOP_WINDOW):
        self.model = model
        self.deterministic = deterministic
        self.window = window
        self.history = deque(maxlen=window)
        self.rng = np.random.default_rng()
        self.loop_breaks = 0

    def reset(self):
        self.history.clear()
        self.loop_breaks = 0

    def predict(self, obs, agent_pos):
        pos = (int(agent_pos[0]), int(agent_pos[1]))
        self.history.append(pos)

        stuck = (len(self.history) >= self.window
                 and len(set(self.history)) <= 2)

        if stuck and len(self.history) >= 2:
            prev = self.history[-2]
            valid = [a for a, (dx, dy) in _ACTION_DIRS.items()
                     if (pos[0] + dx, pos[1] + dy) != prev]
            self.loop_breaks += 1
            return int(self.rng.choice(valid))

        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)

def _eval(model_path, dim, obs, max_steps, n=100, label=""):
    env = gym.make("gymnasium_env/GridWorldCPPV2-v0",
                   size=dim, obs_quantity=obs,
                   max_steps=max_steps, render_mode="rgb_array")
    m = PPO.load(model_path, device="cpu")
    predictor = AntiLoopPredictor(m, deterministic=False)
    full, covs, steps = 0, [], []
    for _ in range(n):
        o, info = env.reset()
        predictor.reset()
        done = trunc = False
        st = 0
        while not done and not trunc:
            a = predictor.predict(o, env.unwrapped._agent_location)
            o, r, done, trunc, info = env.step(a)
            st += 1
        covs.append(info["coverage"])
        steps.append(st)
        if done and not trunc:
            full += 1
    rate = full / n * 100
    avg  = np.mean(covs) * 100
    print(f"  {label:<32} full={full:3d}/{n}  avg={avg:5.2f}%  "
          f"median={np.median(covs)*100:.1f}%  steps={np.mean(steps):.0f}")
    return rate, avg

def _train_stage(name, prev_path, env_fns, timesteps, lr, ent, desc, first=False):
    vec = _make_vec(env_fns)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir    = f"log/ppo_{name}_{tag}"
    model_path = f"data/ppo_{name}_{tag}.zip"

    if prev_path is None:
        m = PPO("MultiInputPolicy", vec,
                n_steps=512, batch_size=128, n_epochs=10,
                learning_rate=lr, ent_coef=ent,
                clip_range=0.2, vf_coef=0.5, device="cpu", verbose=1)
    else:
        m = PPO.load(prev_path, env=vec, device="cpu")
        m.learning_rate = lr
        m.ent_coef      = ent

    logger = configure(log_dir, ["stdout", "csv"])
    m.set_logger(logger)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {desc}")
    print(f"  timesteps={timesteps:,}  lr={lr}  ent_coef={ent}")
    print(f"{'='*60}\n")

    m.learn(total_timesteps=timesteps, reset_num_timesteps=first)
    m.save(model_path)
    print(f"\n  Modelo salvo → {model_path}\n")
    return model_path

# ─── Curriculum completo ──────────────────────────────────────────────────────

def train():
    check_env(gym.make("gymnasium_env/GridWorldCPPV2-v0",
                       size=5, obs_quantity=3, max_steps=200, render_mode="rgb_array"))

    prev = None
    t0   = datetime.now()

    # ── Stage 1: 5×5 — aprende o básico
    prev = _train_stage(
        "cpp_s1_5x5", prev,
        [_make(5, 3, 200)] * 4,
        timesteps=1_500_000, lr=3e-4, ent=0.05, first=True,
        desc="4×5x5 — aprende o básico de CPP",
    )
    print("  Avaliação stage 1:")
    _eval(prev, 5,  3,  200, label="5×5")
    _eval(prev, 10, 12, 500, label="10×10 zero-shot")

    # ── Stage 2: 50/50 — previne catastrophic forgetting ao introduzir 10×10
    prev = _train_stage(
        "cpp_s2_mixed_5050", prev,
        [_make(5,3,200), _make(5,3,200), _make(10,12,500), _make(10,12,500)],
        timesteps=3_000_000, lr=1e-4, ent=0.03,
        desc="2×5x5 + 2×10x10 — introdução gradual do 10×10",
    )
    print("  Avaliação stage 2:")
    _eval(prev, 5,  3,  200, label="5×5")
    _eval(prev, 10, 12, 500, label="10×10")

    # ── Stage 3: 75% 10×10, alta entropia — quebra loops determinísticos
    prev = _train_stage(
        "cpp_s3_10x10_aggressive", prev,
        [_make(5,3,200), _make(10,12,500), _make(10,12,500), _make(10,12,500)],
        timesteps=4_000_000, lr=1e-4, ent=0.08,
        desc="3×10x10 + 1×5x5 — foco agressivo em 10×10",
    )
    print("  Avaliação stage 3:")
    _eval(prev, 5,  3,  200, label="5×5")
    _eval(prev, 10, 12, 500, label="10×10")

    # ── Stage 4: consolidação 5×5 + 10×10
    prev = _train_stage(
        "cpp_s4_consolidation", prev,
        [_make(5,3,200), _make(5,3,200), _make(10,12,500), _make(10,12,500)],
        timesteps=2_000_000, lr=3e-5, ent=0.03,
        desc="2×5x5 + 2×10x10 — consolidação, LR reduzido",
    )
    print("  Avaliação stage 4:")
    _eval(prev, 5,  3,  200, label="5×5")
    _eval(prev, 10, 12, 500, label="10×10")

    # ── Stage 5: introdução 20×20
    prev = _train_stage(
        "cpp_s5_20x20_intro", prev,
        [_make(5,3,200), _make(10,12,500), _make(20,48,2000), _make(20,48,2000)],
        timesteps=5_000_000, lr=1e-4, ent=0.05,
        desc="2×20x20 + 1×10x10 + 1×5x5 — generalização para grid grande",
    )
    print("  Avaliação stage 5:")
    _eval(prev, 5,  3,  200,  label="5×5")
    _eval(prev, 10, 12, 500,  label="10×10")
    _eval(prev, 20, 48, 2000, label="20×20")

    # ── Stage 6: ajuste fino final
    prev = _train_stage(
        "cpp_s6_final", prev,
        [_make(5,3,200), _make(10,12,500), _make(10,12,500), _make(20,48,2000)],
        timesteps=2_000_000, lr=2e-5, ent=0.02,
        desc="final: 2×10x10 + 1×20x20 + 1×5x5",
    )

    print(f"\n{'='*60}")
    print("  AVALIAÇÃO FINAL (100 episódios)")
    print(f"{'='*60}")
    _eval(prev, 5,  3,  200,  label="5×5")
    _eval(prev, 10, 12, 500,  label="10×10")
    _eval(prev, 20, 48, 2000, label="20×20")
    print(f"\nTreinamento completo em {(datetime.now()-t0).total_seconds()/60:.1f} min")
    print(f"Modelo final: {prev}")

# ─── Continuation: stages 7-9 (10×10 > 90% avg, 20×20 ~80%) ─────────────────

def continue_training():
    models = sorted(glob.glob("data/ppo_*.zip"))
    if not models:
        print("Nenhum modelo encontrado em data/. Rode primeiro: python train_grid_world_cpp_v2.py train")
        sys.exit(1)

    print("Modelos disponíveis:")
    for i, p in enumerate(models):
        print(f"  [{i}] {p}")
    idx = input(f"Escolha o índice [{len(models)-1}]: ").strip()
    prev = models[int(idx)] if idx else models[-1]
    print(f"\nIniciando de: {prev}\n")

    t0 = datetime.now()

    # ── Stage 7: 10×10 intensive — quebra loops, alta entropia
    prev = _train_stage(
        "cpp_s7_10x10_intensive", prev,
        [_make(5,3,200), _make(10,12,500), _make(10,12,500), _make(10,12,500)],
        timesteps=5_000_000, lr=5e-5, ent=0.07,
        desc="1×5x5 + 3×10x10 — foco intensivo 10×10, alta entropia",
    )
    print("  Avaliação stage 7:")
    _eval(prev, 5,  3,  200, label="5×5")
    _eval(prev, 10, 12, 500, label="10×10")

    # ── Stage 8: 20×20 introduction
    prev = _train_stage(
        "cpp_s8_20x20_intro", prev,
        [_make(5,3,200), _make(10,12,500), _make(20,48,2000), _make(20,48,2000)],
        timesteps=5_000_000, lr=5e-5, ent=0.05,
        desc="1×5x5 + 1×10x10 + 2×20x20 — introdução 20×20",
    )
    print("  Avaliação stage 8:")
    _eval(prev, 5,  3,  200,  label="5×5")
    _eval(prev, 10, 12, 500,  label="10×10")
    _eval(prev, 20, 48, 2000, label="20×20")

    # ── Stage 9: final fine-tune all sizes
    prev = _train_stage(
        "cpp_s9_final_large", prev,
        [_make(5,3,200), _make(10,12,500), _make(10,12,500), _make(20,48,2000)],
        timesteps=2_000_000, lr=1e-5, ent=0.02,
        desc="1×5x5 + 2×10x10 + 1×20x20 — ajuste fino final",
    )

    print(f"\n{'='*60}")
    print("  AVALIAÇÃO FINAL (100 episódios)")
    print(f"{'='*60}")
    _eval(prev, 5,  3,  200,  label="5×5")
    _eval(prev, 10, 12, 500,  label="10×10")
    _eval(prev, 20, 48, 2000, label="20×20")
    print(f"\nContinuação completa em {(datetime.now()-t0).total_seconds()/60:.1f} min")
    print(f"Modelo final: {prev}")

# ─── Test ─────────────────────────────────────────────────────────────────────

def test(dim):
    cfg = {5: (3, 200), 10: (12, 500), 20: (48, 2000)}
    obs_q, max_steps = cfg.get(dim, (3, 200))

    models = sorted(glob.glob("data/ppo_*.zip"))
    if not models:
        print("Nenhum modelo encontrado em data/. Rode primeiro: python train_grid_world_cpp_v2.py train")
        sys.exit(1)

    print("Modelos disponíveis:")
    for i, p in enumerate(models):
        print(f"  [{i}] {p}")
    idx = input(f"Escolha o índice [{len(models)-1}]: ").strip()
    model_path = models[int(idx)] if idx else models[-1]

    print(f"\nAvaliando {model_path} no grid {dim}×{dim} (100 episódios)...")
    model = PPO.load(model_path, device="cpu")
    env   = gym.make("gymnasium_env/GridWorldCPPV2-v0",
                     size=dim, obs_quantity=obs_q,
                     max_steps=max_steps, render_mode="rgb_array")
    predictor = AntiLoopPredictor(model, deterministic=False)

    full, covs, steps_list, breaks_list = 0, [], [], []
    for ep in range(100):
        obs, info = env.reset()
        predictor.reset()
        done = trunc = False
        steps = 0
        while not done and not trunc:
            action = predictor.predict(obs, env.unwrapped._agent_location)
            obs, r, done, trunc, info = env.step(action)
            steps += 1
        covs.append(info["coverage"])
        steps_list.append(steps)
        breaks_list.append(predictor.loop_breaks)
        if done and not trunc:
            full += 1
            print(f"  Ep {ep+1:3d}: COBERTURA COMPLETA em {steps} passos "
                  f"(loop-breaks={predictor.loop_breaks})")
        else:
            print(f"  Ep {ep+1:3d}: cobertura {info['coverage']:.1%} em {steps} passos "
                  f"(loop-breaks={predictor.loop_breaks})")

    print(f"\n{'─'*50}")
    print(f"  Full Coverage Rate : {full}/100  ({full}%)")
    print(f"  Coverage Média     : {np.mean(covs)*100:.2f}%")
    print(f"  Coverage Mediana   : {np.median(covs)*100:.2f}%")
    print(f"  Steps Médios       : {np.mean(steps_list):.1f}")
    print(f"  Loop-breaks/ep     : {np.mean(breaks_list):.1f}")
    print(f"{'─'*50}")

# ─── Run ──────────────────────────────────────────────────────────────────────

def run(dim):
    cfg = {5: (3, 200), 10: (12, 500), 20: (48, 2000)}
    obs_q, max_steps = cfg.get(dim, (3, 200))

    models = sorted(glob.glob("data/ppo_*.zip"))
    if not models:
        print("Nenhum modelo encontrado.")
        sys.exit(1)

    print("Modelos disponíveis:")
    for i, p in enumerate(models):
        print(f"  [{i}] {p}")
    idx = input(f"Escolha o índice [{len(models)-1}]: ").strip()
    model_path = models[int(idx)] if idx else models[-1]

    print(f"\nVisualizando {model_path} no grid {dim}×{dim}...")
    model = PPO.load(model_path, device="cpu")
    env   = gym.make("gymnasium_env/GridWorldCPPV2-v0",
                     size=dim, obs_quantity=obs_q,
                     max_steps=max_steps, render_mode="human")
    predictor = AntiLoopPredictor(model, deterministic=False)

    obs, info = env.reset()
    predictor.reset()
    done = trunc = False
    steps = 0
    total_r = 0.0
    while not done and not trunc:
        action = predictor.predict(obs, env.unwrapped._agent_location)
        obs, r, done, trunc, info = env.step(action)
        total_r += r
        steps += 1
        print(f"  Step {steps:4d} | {_print_action(action):5s} | "
              f"cov={info['coverage']:.1%} | r={r:+.2f}")

    print(f"\nFim | cov={info['coverage']:.1%} | steps={steps} | reward={total_r:.2f} "
          f"| loop-breaks={predictor.loop_breaks}")
    env.close()

# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "continue", "test", "run"):
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    dim  = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if mode == "train":
        train()
    elif mode == "continue":
        continue_training()
    elif mode == "test":
        test(dim)
    elif mode == "run":
        run(dim)
