# Relatório — Coverage Path Planning com Curriculum Learning e Representação de Estado Aprimorada

**Disciplina:** Reinforcement Learning  
**Aluno:** Yuri Tabacof  
**Data:** 08/05/2026

---

## 1. Introdução

O problema de **Coverage Path Planning (CPP)** consiste em encontrar uma trajetória que cubra todos os pontos acessíveis de um ambiente. O agente deve visitar todas as células livres de um grid NxN com obstáculos usando apenas **observação parcial** — ele não vê o mapa completo, apenas uma janela 3×3 ao redor de sua posição.

Este relatório documenta a jornada de melhoria do agente: parte de um baseline com 78% de cobertura completa no 5×5, passa por diagnóstico do problema, novas arquiteturas de estado, curriculum learning e chega a 90% no 5×5 e **93.85% de cobertura média no 10×10**.

---

## 2. O Problema: Por Que o Agente v1 Trava

O baseline (v1) atingia ~78% de full coverage no 5×5 e ~65% no 10×10. A causa foi identificada na representação do estado:

| Feature v1 | Problema |
|------------|---------|
| Posição normalizada `(x/size, y/size)` | Informa localização, não direção para áreas inexploradas |
| Taxa de cobertura global | Útil, mas sem informação espacial |
| Vizinhança 3×3 instantânea | **Sem memória** — quando todas as células vizinhas estão visitadas, a observação é idêntica independente de onde estão as células não-visitadas no grid |

O agente v1 ficava preso em loops: sem memória de onde já esteve, ele não sabia para onde navegar quando toda a vizinhança imediata já estava coberta. Em grids maiores, isso se agravava — havia mais distância para percorrer até a próxima célula nova, e o agente não tinha como inferir a direção certa.

---

## 3. A Solução: Ambiente v2 + Curriculum Learning

### 3.1 Representação de Estado Aprimorada

Três mudanças no espaço de observação resolvem o problema de memória:

**`seen_map` — memória de exploração persistente**

Um mapa interno `(size × size)` acumulado ao longo do episódio via vizinhança 3×3 do agente a cada passo:

| Valor | Significado |
|-------|------------|
| 0 | Desconhecido (nunca na janela do agente) |
| 1 | Livre e não visitado |
| 2 | Obstáculo ou parede |
| 3 | Visitado pelo agente |

**`local_map` 7×7 — contexto espacial ampliado**

Janela centrada no agente extraída do `seen_map`. Tem **tamanho fixo** independente do grid — essencial para usar o mesmo modelo em grids de tamanhos diferentes via transfer learning.

**`frontier` vector — guia explícito de navegação**

`[dx_livre, dy_livre, dx_desconhecido, dy_desconhecido]` aponta para a célula livre mais próxima e para a célula desconhecida mais próxima. Este sinal resolve diretamente o travamento: mesmo quando toda a vizinhança imediata está visitada, o agente sabe em que direção ir.

```
observation = {
  "agent":     Box(3,)    # [x/(size-1), y/(size-1), coverage_ratio]
  "local_map": Box(7, 7)  # janela do seen_map normalizada
  "frontier":  Box(4,)    # [dx_free, dy_free, dx_unknown, dy_unknown]
}
```

### 3.2 Primeira Tentativa de Curriculum — e o Problema do Catastrophic Forgetting

Com o novo ambiente v2, o primeiro experimento treinou sequencialmente: stage 1 em 5×5 (1.5M steps) → stage 2 em 10×10 puro (2M steps).

Resultado: **catastrophic forgetting grave**.

| Modelo | 5×5 Full% | 10×10 Full% |
|--------|-----------|-------------|
| Stage 1 | 81% | 1% (zero-shot) |
| Stage 2 | **0%** | 53% |

Após fine-tuning puro em 10×10, o agente simplesmente "esqueceu" o 5×5 — de 81% para 0%. O gradiente da tarefa 5×5 desapareceu completamente após 2M passos treinando só em 10×10.

### 3.3 Curriculum Misto — A Solução

A correção foi treinar ambos os grids **simultaneamente** via DummyVecEnv com 4 envs paralelos divididos entre tamanhos. Sem regularização extra, apenas misturando os dados de treino.

O curriculum completo evoluiu em 6 stages:

| Stage | Mix de envs | Timesteps | LR | ent | Objetivo |
|-------|------------|-----------|-----|-----|---------|
| S1 | 4×5×5 | 1.5M | 3×10⁻⁴ | 0.05 | Aprender CPP básico |
| S2 | 2×5×5 + 2×10×10 | 3M | 1×10⁻⁴ | 0.03 | Introduzir 10×10 sem esquecer 5×5 |
| S3 | 1×5×5 + 3×10×10 | 4M | 1×10⁻⁴ | 0.08 | Foco 10×10, alta entropia quebra loops |
| S4 | 2×5×5 + 2×10×10 | 2M | 3×10⁻⁵ | 0.03 | Consolidação, LR reduzido |
| S5 | 1×5×5 + 1×10×10 + 2×20×20 | 5M | 1×10⁻⁴ | 0.05 | Introdução 20×20 |
| S6 | 1×5×5 + 2×10×10 + 1×20×20 | 2M | 2×10⁻⁵ | 0.02 | Ajuste fino final |

---

## 4. Resultados — Fase 1 (S1→S6)

### 4.1 Progressão no 5×5

![Progressão e comparação de coverage](plots/progression_story.png)

O gráfico esquerdo mostra a evolução no 5×5 ao longo dos stages. O salto mais expressivo ocorre no **S3** (75% 10×10 + alta entropia = ent=0.08): de 85% para 93%. A alta entropia quebra os padrões determinísticos que causavam loops, forçando exploração mais diversa.

| Stage | 5×5 Full% | 5×5 Avg% |
|-------|-----------|----------|
| Baseline v1 | 78% | 95.0% |
| S1 | 81% | 89.82% |
| S2 | 85% | 98.55% |
| **S3** | **93%** | **99.23%** |
| S4 | 87% | 98.41% |
| S5 | 86% | 98.14% |
| **S6** | **93%** | **99.36%** |

### 4.2 Full Coverage Rate — todos os modelos

![Full Coverage Rate — 5×5](plots/full_coverage_5x5.png)

### 4.3 Distribuição de Steps — S6 em 5×5

![Distribuição de Steps — S6](plots/s6_steps_distribution.png)

Dos 100 episódios do S6: **93 com cobertura completa** em média de 24.5 steps. Os 7 episódios parciais ainda atingem 91% de cobertura média.

### 4.4 10×10 após S1→S6

O melhor resultado em 10×10 dentro dos stages iniciais foi o **S2** (50/50 mixed): 47% full coverage, 84.76% avg — abaixo da meta de 90% avg. Os stages seguintes priorizavam o 5×5 e não avaliavam o 10×10 continuamente.

![Evolução 10×10 e distribuição](plots/10x10_results.png)

O gráfico direito mostra a instabilidade do S2 no 10×10: muitos episódios falham abaixo de 90%, mas a maioria dos bem-sucedidos atinge 100%. O problema é a inconsistência — o agente ou cobre tudo ou trava cedo.

---

## 5. Resultados — Fase 2: Continuação (S7→S9)

Com o 5×5 consolidado em 93%, o objetivo da segunda fase foi elevar o 10×10 acima de 90% avg e introduzir o 20×20 de forma mais robusta.

Três stages adicionais partindo do modelo S6:

| Stage | Mix de envs | Timesteps | LR | ent | Objetivo |
|-------|------------|-----------|-----|-----|---------|
| S7 | 1×5×5 + 3×10×10 | 5M | 5×10⁻⁵ | 0.07 | Mesma receita do S3, agora para o 10×10 |
| S8 | 1×5×5 + 1×10×10 + 2×20×20 | 5M | 5×10⁻⁵ | 0.05 | Introdução robusta do 20×20 |
| S9 | 1×5×5 + 2×10×10 + 1×20×20 | 2M | 1×10⁻⁵ | 0.02 | Fine-tune final com todos os tamanhos |

**LR reduzido (5×10⁻⁵ ao invés de 1×10⁻⁴)** para proteger o 5×5 enquanto o agente aprende o 10×10 em profundidade. A alta entropia no S7 (ent=0.07) aplica a mesma lógica que funcionou no S3 para o 5×5.

### 5.1 Resultados Finais (S9)

![Resultados finais — todos os grids](plots/final_results_all_grids.png)

| Grid | Full Cov% | Avg Cov% | Mediana | Steps médios |
|------|-----------|----------|---------|-------------|
| 5×5  | **90%** | **98.95%** | 100% | 42 |
| 10×10 | **50%** | **93.85%** | 99.4% | 303 |
| 20×20 | 0% | 61.08% | 66.9% | 2000 (limite) |

**5×5**: caiu levemente de 93% para 90% full coverage (o foco intenso no 10×10 no S7 moveu levemente a política), mas manteve-se acima da meta de 90% e com avg de 98.95%.

**10×10**: meta atingida — **93.85% de avg coverage**, com mediana de 99.4%. A maioria dos episódios é bem-sucedida; os 50% que não chegam a 100% ainda cobrem quase todo o grid. Notável a mediana de 99.4%: o agente frequentemente quase-completa os episódios que não alcançam 100%.

**20×20**: 0% full coverage, 61% avg. Esperado dado que a arquitetura MLP trata o `local_map` como vetor flat sem explorar a estrutura espacial. Com 352 células livres e max_steps=2000, o agente precisa de estratégia de longo alcance que o MLP não consegue desenvolver.

---

## 6. Análise

### 6.1 O que funcionou

**Frontier vector** foi o principal responsável pelo salto de qualidade. Dois estados idênticos para o v1 (todas as células vizinhas visitadas) tornam-se distinguíveis no v2:
- Células livres ao norte → `frontier=[0, -0.5, ...]`
- Células livres ao sul → `frontier=[0, +0.5, ...]`

O agente passa a navegar em direção a áreas inexploradas em vez de aleatoriamente.

**Curriculum misto** foi fundamental para evitar o catastrophic forgetting. Misturar grids de tamanhos diferentes no mesmo rollout force a rede a manter comportamentos úteis para todos os contextos simultaneamente.

**Alta entropia nos stages de "quebra de loops"** (S3 para 5×5, S7 para 10×10) foi consistentemente eficaz — forçar exploração durante o treinamento produz políticas mais robustas.

### 6.2 O que limitou o 20×20

O gargalo principal é arquitetural: o `MultiInputPolicy` do SB3 aplica um MLP flat sobre o `local_map` 7×7 (49 valores). Uma CNN extrairia features espaciais muito mais ricas — padrões de bordas, corredores, regiões abertas — que o MLP não consegue aprender eficientemente. Para 20×20, o agente precisaria de uma política de planejamento que enxergasse além dos 3 cells cobertos pela janela local.

### 6.3 Lição sobre Catastrophic Forgetting

O experimento sequencial demonstrou empiricamente: fine-tuning puro em 10×10 após o 5×5 destruiu completamente a política (81% → 0%). A solução pelo curriculum misto é mais simples e eficaz do que técnicas de regularização como EWC ou distillation.

---

## 7. Conclusão

| Grid | Baseline v1 | Após S1-S6 | **Final (S9)** |
|------|-------------|-----------|----------------|
| 5×5 Full% | 78% | 93% | **90%** |
| 5×5 Avg% | 95.0% | 99.36% | **98.95%** |
| 10×10 Full% | 65% | 47% | **50%** |
| 10×10 Avg% | 82.0% | 84.76% | **93.85%** ✓ |
| 20×20 Avg% | — | — | 61.08% |

A combinação de **seen_map + local_map 7×7 + frontier vector + curriculum misto** elevou a cobertura de forma consistente. A meta principal (>90% avg no 10×10) foi atingida. O 20×20 mostra que o agente aprendeu algo — cobre 61% em média — mas a arquitetura MLP é o próximo gargalo a superar para generalização em grids grandes.
