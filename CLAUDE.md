# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a reinforcement learning training repository for the WF-TRON robot (a quadruped with wheeled feet), built on top of the `mjlab` framework and using RSL-RL's PPO implementation.

## Common Commands

### Installation
```shell
uv sync
```

### Training
```shell
uv run python scripts/rsl_rl/train.py Mjlab-WF-Tron
```

### Evaluation
```shell
uv run python scripts/rsl_rl/play.py Mjlab-WF-Tron --checkpoint-file /path/to/model.pt
```

### Key Training Parameters
Located in `src/tron1_rl_mjlab/tasks/cfg/wf_tron_rl_cfg.py`:
- `max_iterations`: Maximum training iterations (default: 15000)
- `save_interval`: Checkpoint save interval in iterations (default: 200)
- `num_steps_per_env`: Steps per environment per iteration (default: 24)

## Architecture

### Task Registration
Tasks are registered via `register_mjlab_task()` in `src/tron1_rl_mjlab/__init__.py`. The task "Mjlab-WF-Tron" is defined by three factory functions:
- `make_wf_tron_env_cfg()`: Environment configuration (scene, observations, actions, commands, rewards, events, terminations, curriculum)
- `make_wf_tron_play_env_cfg()`: Play/evaluation environment (reduced to 8 envs)
- `make_wf_tron_rl_cfg()`: RL runner configuration (algorithm, network architecture, training params)

### Configuration Structure
- `wf_tron_env_cfg.py`: Environment factory functions (scene, observations, actions, commands, rewards, events, terminations, curriculum)
- `wf_tron_rl_cfg.py`: RL training configuration (PPO algorithm, actor/critic networks, max iterations, save interval)
- `terrain_cfg.py`: Terrain configuration (plane or procedural terrain with boxes, rough surfaces, slopes)
- `wf_tron.py` (in assets): Robot XML model, actuator configurations, initial state

### MDP Components (`tasks/mdp/`)
- `observations.py`: Observation functions (base velocity, joint positions, gravity projection, etc.)
- `rewards.py`: Reward functions (tracking, safety, penalties)
- `commands.py`: Command generators (pose commands, velocity commands)
- `events.py`: Event handlers (domain randomization, reset handlers)
- `terminations.py`: Termination conditions (timeouts, bad orientation, bad height)
- `curriculums.py`: Curriculum learning strategies

### Training Flow
1. `train.py` loads env and RL configs from the registry
2. Creates `ManagerBasedRlEnv` with all MDP components
3. Wraps with `RslRlVecEnvWrapper`
4. Creates `MjlabOnPolicyRunner` and calls `runner.learn(num_learning_iterations=...)`
5. Checkpoints saved to `logs/rsl_rl/{experiment_name}/{timestamp}/model_{iter}.pt`

### Logging
Weights & Biases is used for experiment tracking. Configure via `wandb_project` and `experiment_name` in the RL config.
