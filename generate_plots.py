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

# 5×5 — full coverage rate e avg coverage por modelo (todos os stages)
models_5x5 = [
    "Baseline v1\n(enunciado)",
    "v2b s2\nmixed",
    "s1\n5x5 only",
    "s2\n50/50",
    "s3\n75%10x10",
    "s4\nconsolid.",
    "s5\n+20x20",
    "s6\nfinal",
    "s9\nfinal large",
]
full_5x5 = [78, 88, 85, 85, 93, 87, 86, 93, 90]
avg_5x5  = [95.0, 97.64, 98.55, 98.55, 99.23, 98.41, 98.14, 99.36, 98.95]

# 10×10 — por modelo
models_10x10 = ["Baseline v1\n(enunciado)", "s2\n50/50", "s9\nfinal large"]
full_10x10   = [65, 47, 50]
avg_10x10    = [82.0, 84.76, 93.85]

# Progressão completa (5×5) — stages 1-6
stages_orig       = ["Baseline\nv1", "S1\n5×5", "S2\n50/50", "S3\n75%10×10", "S4\nconsolid.", "S5\n+20×20", "S6\nfinal"]
prog_full_orig    = [78,             81,         85,           93,              87,              86,           93]
prog_avg_orig     = [95.0,           89.82,      98.55,        99.23,           98.41,           98.14,        99.36]

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

# ── Fig 1: Progressão completa — storytelling (todas as fases) ────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Esquerda: 5×5 progression
ax1.plot(stages_orig, prog_full_orig, "o-", color=BLUE,  linewidth=2, markersize=7, label="Full Cov %")
ax1.plot(stages_orig, prog_avg_orig,  "s--", color=GREEN, linewidth=2, markersize=7, label="Avg Cov %")
for i, (f, a) in enumerate(zip(prog_full_orig, prog_avg_orig)):
    ax1.annotate(f"{f}%", (stages_orig[i], f), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color=BLUE)
ax1.axhline(90, color=RED, linestyle=":", linewidth=1.2, alpha=0.8, label="Meta 90%")
ax1.axvspan(-0.5, 0.5, alpha=0.08, color=GRAY, label="Baseline v1")
ax1.set_ylim(70, 110)
ax1.set_ylabel("Cobertura (%)")
ax1.set_title("Progressão — Grid 5×5 (S1→S6)")
ax1.legend(frameon=False, fontsize=9)

# Direita: comparação 5×5 vs 10×10 avg nos pontos-chave
key_models  = ["Baseline v1", "S2 mixed\n(50/50)", "S6 final", "S9 final\nlarge"]
avg_5_key   = [95.0,          98.55,                99.36,       98.95]
avg_10_key  = [82.0,          84.76,                None,        93.85]

x = np.arange(len(key_models))
w = 0.35
b1 = ax2.bar(x - w/2, avg_5_key, w, label="5×5 avg %", color=BLUE, alpha=0.9)
avg_10_plot = [v if v is not None else 0 for v in avg_10_key]
b2 = ax2.bar(x + w/2, avg_10_plot, w, label="10×10 avg %", color=ORANGE, alpha=0.9)
# mark S6 10×10 as N/A
ax2.text(x[2] + w/2, 2, "N/A", ha="center", va="bottom", fontsize=8, color=GRAY)
ax2.axhline(90, color=RED, linestyle="--", linewidth=1.2, alpha=0.7, label="Meta 90%")
ax2.set_xticks(x); ax2.set_xticklabels(key_models, fontsize=9)
ax2.set_ylim(0, 110)
ax2.set_ylabel("Avg Coverage (%)")
ax2.set_title("Avg Coverage — pontos-chave")
ax2.legend(frameon=False, fontsize=9)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                 f"{h:.0f}", ha="center", va="bottom", fontsize=8)
fig.suptitle("Evolução do agente CPP — do Baseline ao Curriculum Learning", fontsize=12)
plt.tight_layout()
plt.savefig("plots/progression_story.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: plots/progression_story.png")

# ── Fig 2: Full coverage rate — todos os modelos 5×5 ─────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
colors = [GRAY, BLUE, BLUE, BLUE, GREEN, BLUE, BLUE, GREEN, PURPLE]
bars = ax.bar(models_5x5, full_5x5, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(90, color=RED, linestyle="--", linewidth=1.2, label="Meta 90%")
for bar, val in zip(bars, full_5x5):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Full Coverage Rate (%)")
ax.set_title("Full Coverage Rate — Grid 5×5 (100 episódios)")
ax.set_ylim(0, 108)
baseline_patch = mpatches.Patch(color=GRAY,   label="Baseline v1")
best_patch     = mpatches.Patch(color=GREEN,  label="Melhor (S3/S6)")
cont_patch     = mpatches.Patch(color=PURPLE, label="S9 (continuação)")
ax.legend(handles=[baseline_patch, best_patch, cont_patch,
                   mpatches.Patch(color=RED, label="Meta 90%")],
          frameon=False)
plt.tight_layout()
plt.savefig("plots/full_coverage_5x5.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: plots/full_coverage_5x5.png")

# ── Fig 3: 10×10 avg coverage — evolução ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

x2 = np.arange(len(models_10x10))
w = 0.38
bar_full = ax1.bar(x2 - w/2, full_10x10, w, label="Full Cov %",  color=BLUE,   alpha=0.9)
bar_avg  = ax1.bar(x2 + w/2, avg_10x10,  w, label="Avg Cov %",   color=ORANGE, alpha=0.9)
ax1.axhline(90, color=RED, linestyle="--", linewidth=1.2, alpha=0.8, label="Meta avg 90%")
ax1.set_xticks(x2); ax1.set_xticklabels(models_10x10, fontsize=10)
ax1.set_ylim(0, 110)
ax1.set_title("Grid 10×10 — evolução por modelo")
ax1.set_ylabel("Cobertura (%)")
ax1.legend(frameon=False, fontsize=9)
for bar in list(bar_full) + list(bar_avg):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Direita: 10×10 coverage por episódio — s2
ep = np.arange(1, 101)
colors_ep = [GREEN if c == 100 else (ORANGE if c >= 90 else RED) for c in cov_10x10_s2]
ax2.bar(ep, cov_10x10_s2, color=colors_ep, width=1.0, edgecolor="none")
ax2.axhline(90, color=RED, linestyle="--", linewidth=1, alpha=0.7)
ax2.set_xlabel("Episódio")
ax2.set_ylabel("Cobertura (%)")
ax2.set_title("Cobertura por episódio — s2_mixed, 10×10")
green_p  = mpatches.Patch(color=GREEN,  label="100%")
orange_p = mpatches.Patch(color=ORANGE, label="90-99%")
red_p    = mpatches.Patch(color=RED,    label="< 90%")
ax2.legend(handles=[green_p, orange_p, red_p], frameon=False, fontsize=9)
ax2.set_ylim(0, 110)

fig.suptitle("Grid 10×10 — progresso e distribuição de cobertura", fontsize=12)
plt.tight_layout()
plt.savefig("plots/10x10_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: plots/10x10_results.png")

# ── Fig 4: Distribuição de steps — s6_final em 5×5 ───────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.hist(s6_steps_full, bins=range(20, 42, 2), color=GREEN, edgecolor="white", alpha=0.9)
ax1.axvline(np.mean(s6_steps_full), color=RED, linestyle="--",
            label=f"Média: {np.mean(s6_steps_full):.1f}")
ax1.set_xlabel("Steps para cobertura completa")
ax1.set_ylabel("Frequência")
ax1.set_title(f"Distribuição de Steps — S6, 5×5\n(93 episódios completos)")
ax1.legend(frameon=False)

cov_all = [100.0] * len(s6_steps_full) + s6_cov_partial
ax2.hist(cov_all, bins=[0,50,60,70,80,90,95,100,101],
         color=BLUE, edgecolor="white", alpha=0.9)
ax2.set_xlabel("Cobertura (%)")
ax2.set_ylabel("Frequência")
ax2.set_title("Distribuição de Cobertura — S6, 5×5\n(100 episódios)")
plt.tight_layout()
plt.savefig("plots/s6_steps_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: plots/s6_steps_distribution.png")

# ── Fig 5: Resumo final — todos os grids ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

grids   = ["5×5", "10×10", "20×20"]
full_s6 = [93, None, None]
full_s9 = [90, 50, 0]
avg_s6  = [99.36, None, None]
avg_s9  = [98.95, 93.85, 61.08]

x = np.arange(len(grids))
w = 0.2

# S6 full, S9 full, S6 avg, S9 avg
b1 = ax.bar(x - 1.5*w, [v or 0 for v in full_s6], w, color=BLUE,   alpha=0.6, label="S6 Full Cov %")
b2 = ax.bar(x - 0.5*w, full_s9,                    w, color=BLUE,   alpha=1.0, label="S9 Full Cov %")
b3 = ax.bar(x + 0.5*w, [v or 0 for v in avg_s6],  w, color=GREEN,  alpha=0.6, label="S6 Avg Cov %")
b4 = ax.bar(x + 1.5*w, avg_s9,                     w, color=GREEN,  alpha=1.0, label="S9 Avg Cov %")

ax.axhline(90, color=RED, linestyle="--", linewidth=1.2, alpha=0.8, label="Meta 90%")

for bars in [b1, b2, b3, b4]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8)

# Mark S6 10×10 and 20×20 as N/A
ax.text(x[1] - 1.5*w + w/2, 2, "N/A", ha="center", va="bottom", fontsize=7, color=GRAY)
ax.text(x[2] - 1.5*w + w/2, 2, "N/A", ha="center", va="bottom", fontsize=7, color=GRAY)
ax.text(x[1] + 0.5*w + w/2, 2, "N/A", ha="center", va="bottom", fontsize=7, color=GRAY)
ax.text(x[2] + 0.5*w + w/2, 2, "N/A", ha="center", va="bottom", fontsize=7, color=GRAY)

ax.set_xticks(x); ax.set_xticklabels(grids, fontsize=12)
ax.set_ylim(0, 112)
ax.set_ylabel("Cobertura (%)")
ax.set_title("Resultados Finais — S6 vs S9 por tamanho de grid")
ax.legend(frameon=False, fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig("plots/final_results_all_grids.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: plots/final_results_all_grids.png")

print("\nTodos os plots gerados em plots/")
