"""Gera os plots do relatório. Execute: python generate_plots.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

# ── Paleta ────────────────────────────────────────────────────────────────────
BLUE   = "#2196F3"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
RED    = "#F44336"
GRAY   = "#9E9E9E"
PURPLE = "#9C27B0"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})

# ── Dados ─────────────────────────────────────────────────────────────────────

# 5×5 — full coverage rate e avg coverage por modelo
models_5x5 = [
    "Baseline v1\n(enunciado)",
    "v2b s2\nmixed",
    "s1\n5x5 only",
    "s2\n50/50",
    "s3\n75%10x10",
    "s4\nconsolidação",
    "s5\n+20x20",
    "s6\nfinal",
]
full_5x5 = [78, 88, 85, 85, 93, 87, 86, 93]
avg_5x5  = [95.0, 97.64, 98.55, 98.55, 99.23, 98.41, 98.14, 99.36]

# 10×10 — disponível apenas para alguns modelos
models_10x10 = ["Baseline v1\n(enunciado)", "s2\n50/50"]
full_10x10   = [65, 47]
avg_10x10    = [82.0, 84.76]

# Progressão por stage (5×5)
stages       = ["Baseline\nv1", "S1\n5×5", "S2\n50/50", "S3\n75%10x10", "S4\nconsolid.", "S5\n+20×20", "S6\nfinal"]
prog_full    = [78,             81,         85,           93,              87,              86,           93]
prog_avg     = [95.0,           89.82,      98.55,        99.23,           98.41,           98.14,        99.36]

# Episódios detalhados — s6_final em 5×5 (93 completos, 7 parciais)
s6_steps_full = [
    23,30,25,23,27,25,27,22,21,28,24,22,23,23,26,23,25,24,22,28,
    22,23,27,22,27,23,27,24,30,26,27,27,24,24,21,25,23,25,23,22,
    25,23,22,26,29,25,27,23,28,24,27,22,22,23,23,23,23,26,29,23,
    25,24,26,21,24,27,28,26,29,22,21,28,22,23,25,27,23,23,24,24,
    25,21,27,25,24,24,27,25,25,23,27,22,27
]
s6_cov_partial = [95.5, 95.5, 95.5, 90.9, 95.5, 68.2, 95.5]

# Cobertura por episódio — s2_mixed_5050 em 10×10
cov_10x10_s2 = [
    100,98.9,100,100,100,98.9,96.6,27.3,100,100,100,100,100,100,92.0,
    17.0,44.3,55.7,100,100,98.9,100,100,100,85.2,100,37.5,90.9,100,100,
    100,100,61.4,100,94.3,100,100,100,100,9.1,38.6,98.9,39.8,78.4,69.3,
    100,5.7,37.5,98.9,100,98.9,18.2,100,90.9,100,76.1,100,98.9,96.6,100,
    83.0,100,98.9,98.9,100,76.1,62.5,97.7,98.9,86.4,100,50.0,73.9,100,100,
    98.9,89.8,100,47.7,100,98.9,93.2,100,83.0,19.3,100,100,100,100,94.3,
    38.6,18.2,100,100,18.2,98.9,100,98.9,100,97.7
]

# ── Fig 1: Comparação full coverage rate — 5×5 ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
colors = [GRAY, BLUE, BLUE, BLUE, GREEN, BLUE, BLUE, GREEN]
bars = ax.bar(models_5x5, full_5x5, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(90, color=RED, linestyle="--", linewidth=1.2, label="Meta 90%")
for bar, val in zip(bars, full_5x5):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Full Coverage Rate (%)")
ax.set_title("Full Coverage Rate — Grid 5×5 (100 episódios)")
ax.set_ylim(0, 105)
ax.legend(frameon=False)
baseline_patch = mpatches.Patch(color=GRAY, label="Baseline v1")
new_patch = mpatches.Patch(color=GREEN, label="Melhor resultado v2")
ax.legend(handles=[baseline_patch, new_patch,
                   mpatches.Patch(color=RED, label="Meta 90%", linestyle="--")],
          frameon=False)
plt.tight_layout()
plt.savefig("plots/full_coverage_5x5.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: plots/full_coverage_5x5.png")

# ── Fig 2: Avg coverage e full coverage — ambos os grids ─────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# 5×5
x = np.arange(len(models_5x5))
w = 0.4
b1 = ax1.bar(x - w/2, full_5x5, w, label="Full Cov %", color=BLUE, alpha=0.9)
b2 = ax1.bar(x + w/2, avg_5x5,  w, label="Avg Cov %",  color=GREEN, alpha=0.9)
ax1.set_xticks(x); ax1.set_xticklabels(models_5x5, fontsize=8)
ax1.set_ylim(0, 110); ax1.set_title("Grid 5×5"); ax1.set_ylabel("Cobertura (%)")
ax1.axhline(90, color=RED, linestyle="--", linewidth=1, alpha=0.7)
ax1.legend(frameon=False, fontsize=9)

# 10×10
x2 = np.arange(len(models_10x10))
ax2.bar(x2 - w/2, full_10x10, w, label="Full Cov %", color=BLUE, alpha=0.9)
ax2.bar(x2 + w/2, avg_10x10,  w, label="Avg Cov %",  color=GREEN, alpha=0.9)
ax2.set_xticks(x2); ax2.set_xticklabels(models_10x10, fontsize=9)
ax2.set_ylim(0, 110); ax2.set_title("Grid 10×10"); ax2.set_ylabel("Cobertura (%)")
ax2.axhline(90, color=RED, linestyle="--", linewidth=1, alpha=0.7, label="Meta 90%")
ax2.legend(frameon=False, fontsize=9)
for ax in (ax1, ax2):
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)
fig.suptitle("Full Coverage Rate vs Average Coverage por modelo", fontsize=12)
plt.tight_layout()
plt.savefig("plots/coverage_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: plots/coverage_comparison.png")

# ── Fig 3: Progressão por stage (5×5) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(stages, prog_full, "o-", color=BLUE,  linewidth=2, markersize=7, label="Full Cov %")
ax.plot(stages, prog_avg,  "s--", color=GREEN, linewidth=2, markersize=7, label="Avg Cov %")
for i, (f, a) in enumerate(zip(prog_full, prog_avg)):
    ax.annotate(f"{f}%", (stages[i], f), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9, color=BLUE)
ax.axhline(90, color=RED, linestyle=":", linewidth=1.2, alpha=0.8, label="Meta 90%")
ax.set_ylim(70, 105)
ax.set_ylabel("Cobertura (%)")
ax.set_title("Progressão durante Curriculum Learning — Grid 5×5")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("plots/curriculum_progression.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: plots/curriculum_progression.png")

# ── Fig 4: Distribuição de steps — s6_final em 5×5 ───────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.hist(s6_steps_full, bins=range(20, 42, 2), color=GREEN, edgecolor="white", alpha=0.9)
ax1.axvline(np.mean(s6_steps_full), color=RED, linestyle="--",
            label=f"Média: {np.mean(s6_steps_full):.1f}")
ax1.set_xlabel("Steps para cobertura completa")
ax1.set_ylabel("Frequência")
ax1.set_title(f"Distribuição de Steps — s6_final, 5×5\n(93 episódios com cobertura completa)")
ax1.legend(frameon=False)

cov_all = [100.0] * len(s6_steps_full) + s6_cov_partial
ax2.hist(cov_all, bins=[0,50,60,70,80,90,95,100,101],
         color=BLUE, edgecolor="white", alpha=0.9)
ax2.set_xlabel("Cobertura (%)")
ax2.set_ylabel("Frequência")
ax2.set_title("Distribuição de Cobertura — s6_final, 5×5\n(100 episódios)")
plt.tight_layout()
plt.savefig("plots/s6_steps_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: plots/s6_steps_distribution.png")

# ── Fig 5: Cobertura por episódio — 10×10 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
ep = np.arange(1, 101)
colors_ep = [GREEN if c == 100 else (ORANGE if c >= 90 else RED) for c in cov_10x10_s2]
ax.bar(ep, cov_10x10_s2, color=colors_ep, width=1.0, edgecolor="none")
ax.axhline(100, color=GREEN, linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlabel("Episódio")
ax.set_ylabel("Cobertura (%)")
ax.set_title(f"Cobertura por Episódio — s2_mixed, 10×10  "
             f"(Full Cov: 47/100 | Avg: 84.8%)")
green_p  = mpatches.Patch(color=GREEN,  label="100% (completo)")
orange_p = mpatches.Patch(color=ORANGE, label="90-99%")
red_p    = mpatches.Patch(color=RED,    label="< 90%")
ax.legend(handles=[green_p, orange_p, red_p], frameon=False, fontsize=9)
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig("plots/10x10_coverage_per_episode.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: plots/10x10_coverage_per_episode.png")

print("\nTodos os plots gerados em plots/")
