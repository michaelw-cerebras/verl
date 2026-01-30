# Generalized Knowledge Distillation (GKD) in PPO Trainer

This document describes the GKD (Generalized Knowledge Distillation) integration into verl's PPO/GRPO trainer, enabling flexible combinations of reinforcement learning and teacher-guided distillation.

## Overview

The integrated GKD feature supports three training modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Pure GRPO** | Standard RL without distillation | Original verl behavior |
| **GKD Only** | Pure distillation without RL loss | Knowledge transfer from teacher |
| **GKD + GRPO** | Combined RL and distillation | Best of both worlds |

## Training Modes

### Mode 1: Pure GRPO (Original verl)

Standard reinforcement learning without any teacher involvement.

```yaml
actor_rollout_ref:
  actor:
    use_teacher_kl_loss: false  # Disable GKD
  rollout:
    n: 4  # Multiple rollouts for GRPO advantage estimation
```

**Characteristics:**
- No teacher server required
- On-policy only
- Loss: GRPO policy gradient

---

### Mode 2: GKD Only (Pure Distillation)

Pure knowledge distillation from teacher to student without RL loss. Supports both on-policy (student-generated) and off-policy (ground truth) data.

```yaml
actor_rollout_ref:
  actor:
    use_teacher_kl_loss: true
    gkd_only_mode: true
    teacher_kl_coef: 1.0
  rollout:
    n: 1  # Usually 1 for pure distillation
  teacher:
    server_ip: "127.0.0.1"
    server_port: 15555
    n_server_workers: 1
    temperature: 1.0

trainer:
  enable_off_policy: true    # Optional: enable off-policy data
  gkd_lambda: 0.7            # On/off policy mixing ratio (1.0 = pure on-policy)
```

**Characteristics:**
- Teacher server required
- Supports on-policy and off-policy mixing via `gkd_lambda`
- Loss: Teacher KL only (no RL loss)
- Skips reward computation and critic updates

**Lambda Control:**
- `gkd_lambda = 1.0`: Pure on-policy (student generates responses)
- `gkd_lambda = 0.0`: Pure off-policy (ground truth answers)
- `0 < gkd_lambda < 1`: Disjoint partition mixing

---

### Mode 3: GKD + GRPO (Combined RL and Distillation)

Combines GRPO reinforcement learning with teacher-guided distillation. This mode is on-policy only since GRPO requires multiple rollouts for advantage estimation.

```yaml
actor_rollout_ref:
  actor:
    use_teacher_kl_loss: true
    gkd_only_mode: false
    teacher_kl_coef: 0.5
    # Selection strategy for which responses get teacher KL
    gkd_select_strategy: "all"  # all / random / best / worst
    gkd_select_k: 1             # Number of responses to select (for random/best/worst)
  rollout:
    n: 4  # Multiple rollouts for GRPO
  teacher:
    server_ip: "127.0.0.1"
    server_port: 15555

trainer:
  enable_off_policy: false  # Off-policy not supported with GRPO
```

**Characteristics:**
- Teacher server required
- On-policy only (GRPO needs rollouts)
- Loss: GRPO policy gradient + Teacher KL
- Supports selection strategies for efficient teacher inference

---

## Selection Strategies (GKD + GRPO Mode)

When `rollout.n > 1`, you can control which responses receive teacher KL distillation:

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `all` | Apply teacher KL to all n responses | Maximum training signal (default) |
| `random` | Randomly select k responses | Cost-efficient, unbiased baseline |
| `best` | Select top-k by reward | Reinforce good behavior with teacher guidance |
| `worst` | Select bottom-k by reward | Correct poor behavior with teacher guidance |

**Configuration:**
```yaml
actor_rollout_ref:
  actor:
    gkd_select_strategy: "random"  # all / random / best / worst
    gkd_select_k: 2                # Select 2 out of n responses
```

**Example (n=4, k=2, strategy=best):**
```
Prompt → [resp_0: r=0.3, resp_1: r=0.8, resp_2: r=0.5, resp_3: r=0.2]
                         ↑ selected      ↑ selected
Teacher KL computed only for resp_1 and resp_2 (highest rewards)
GRPO loss computed for all 4 responses
```

---

## Configuration Reference

### Actor Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_teacher_kl_loss` | `false` | Enable GKD teacher distillation |
| `gkd_only_mode` | `false` | Pure distillation (no RL loss) |
| `teacher_kl_coef` | `1.0` | Weight for teacher KL loss |
| `teacher_kl_temperature` | `1.0` | Temperature for KL computation |
| `gkd_select_strategy` | `"all"` | Selection strategy for GRPO+GKD |
| `gkd_select_k` | `1` | Number of responses to select |

### Trainer Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_off_policy` | `false` | Enable off-policy data (GKD only mode) |
| `gkd_lambda` | `1.0` | On/off policy ratio (1.0 = pure on-policy) |
| `data_partition_seed` | `12345` | Seed for disjoint partition |

### Teacher Configuration

| Parameter | Description |
|-----------|-------------|
| `server_ip` | Teacher server IP address |
| `server_port` | Teacher server port |
| `n_server_workers` | Number of teacher workers |
| `temperature` | Sampling temperature for teacher |

### Data Configuration (Off-Policy)

| Parameter | Description |
|-----------|-------------|
| `off_policy_files` | Path to off-policy data (parquet) |
| `off_policy_batch_size` | Batch size for off-policy data |

---

## Usage Examples

### Example 1: Pure GRPO (Baseline)

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name=ppo_trainer \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.use_teacher_kl_loss=false \
  actor_rollout_ref.rollout.n=4 \
  data.train_files=data/train.parquet \
  trainer.total_epochs=2
```

### Example 2: GKD Only (Pure Distillation)

First, start the teacher server:
```bash
cd recipe/gkd/teacher
bash start_server.sh
```

Then run training:
```bash
python3 -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name=ppo_trainer \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.use_teacher_kl_loss=true \
  actor_rollout_ref.actor.gkd_only_mode=true \
  actor_rollout_ref.actor.teacher_kl_coef=1.0 \
  actor_rollout_ref.teacher.server_ip=127.0.0.1 \
  actor_rollout_ref.teacher.server_port=15555 \
  actor_rollout_ref.rollout.n=1 \
  data.train_files=data/train.parquet \
  trainer.total_epochs=2
```

### Example 3: GKD Only with Off-Policy Mixing

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name=ppo_trainer \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.use_teacher_kl_loss=true \
  actor_rollout_ref.actor.gkd_only_mode=true \
  actor_rollout_ref.teacher.server_ip=127.0.0.1 \
  actor_rollout_ref.teacher.server_port=15555 \
  trainer.enable_off_policy=true \
  trainer.gkd_lambda=0.7 \
  data.train_files=data/train.parquet \
  data.off_policy_files=data/ground_truth.parquet \
  trainer.total_epochs=2
```

### Example 4: GKD + GRPO (Combined)

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name=ppo_trainer \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.use_teacher_kl_loss=true \
  actor_rollout_ref.actor.gkd_only_mode=false \
  actor_rollout_ref.actor.teacher_kl_coef=0.5 \
  actor_rollout_ref.teacher.server_ip=127.0.0.1 \
  actor_rollout_ref.teacher.server_port=15555 \
  actor_rollout_ref.rollout.n=4 \
  algorithm.adv_estimator=grpo \
  data.train_files=data/train.parquet \
  trainer.total_epochs=2
```

### Example 5: GKD + GRPO with Selection Strategy

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=config \
  --config-name=ppo_trainer \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.use_teacher_kl_loss=true \
  actor_rollout_ref.actor.gkd_only_mode=false \
  actor_rollout_ref.actor.teacher_kl_coef=0.5 \
  actor_rollout_ref.actor.gkd_select_strategy=best \
  actor_rollout_ref.actor.gkd_select_k=2 \
  actor_rollout_ref.teacher.server_ip=127.0.0.1 \
  actor_rollout_ref.teacher.server_port=15555 \
  actor_rollout_ref.rollout.n=4 \
  algorithm.adv_estimator=grpo \
  data.train_files=data/train.parquet \
  trainer.total_epochs=2
```

---

## Metrics

The following metrics are logged during GKD training:

| Metric | Description |
|--------|-------------|
| `actor/teacher_kl_loss` | Teacher KL divergence loss |
| `actor/teacher_kl_coef` | Teacher KL loss coefficient |
| `gkd/select_ratio` | Ratio of responses selected for teacher KL |
| `gkd/select_strategy` | Current selection strategy |
| `data/is_on_policy` | Whether current batch is on-policy (1.0) or off-policy (0.0) |
| `timing/teacher_knowledge` | Time spent getting teacher knowledge |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PPO Trainer                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Rollout   │───▶│   Teacher   │───▶│    Actor    │         │
│  │   (vLLM)    │    │   Client    │    │   Update    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│        │                  │                   │                  │
│        ▼                  ▼                   ▼                  │
│   responses          teacher_logps      GRPO + KL loss          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Mode Selection:                                                 │
│  • use_teacher_kl=false        → Pure GRPO                      │
│  • use_teacher_kl=true, gkd_only=true  → GKD Only               │
│  • use_teacher_kl=true, gkd_only=false → GKD + GRPO             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Notes

1. **Teacher Server**: Must be running before training starts. See `recipe/gkd/teacher/` for setup instructions.

2. **Off-Policy Data**: When using off-policy mode, ensure the off-policy dataset has the same format as the training data, with `question` and `answer` fields.

3. **Selection Strategy**: The `best`/`worst` strategies require reward computation, so they only work in GKD+GRPO mode (not GKD only mode).

4. **Memory Considerations**: When using `gkd_select_strategy=all` with large `rollout.n`, teacher inference cost scales linearly. Consider using `random` or `best`/`worst` to reduce cost.

5. **Compatibility**: GKD integration works with both FSDP and Megatron backends for actor training.

---

## References

- [On-Policy Distillation Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [GKD Recipe README](../../recipe/gkd/README.md)
- [GRPO Algorithm](./grpo.md)
