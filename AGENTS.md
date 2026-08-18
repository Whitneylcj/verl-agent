# Repository Guidelines

## Project Scope and Layout

This fork develops Agentic RL methods on `verl-agent`. Core code lives in `verl/`; environments, memory, rollout, and rewards live in `agent_system/`. Put research logic and configs in `recipe/<method>/`, launchers in `examples/<method>/`, tests in `tests/`, and documentation in `docs/`. Prefer thin framework adapters over broad rewrites.

## Local and Remote Responsibilities

**Write locally, train remotely, synchronize through Git.** macOS is for implementation, static checks, Git, and lightweight CPU tests. Do not install `requirements.txt` merely to satisfy imports or run CUDA, vLLM, FlashAttention, distributed FSDP, full rollouts, or RL training locally.

GPU dependencies, Ray, agent environments, smoke rollouts, training, and evaluation belong on the remote Linux host. Local success does not prove the remote pipeline works; never report an unexecuted GPU result.

## Development and Git Workflow

Use `origin` for the research fork and `upstream` for `langfengQ/verl-agent`; work on dedicated branches. Follow: local edit/test → commit/push → remote pull → smoke test → experiment. Prefer Git over ad-hoc file copies. Record `git rev-parse HEAD` before important runs. Never commit datasets, caches, checkpoints, rollout dumps, W&B data, or large logs.

## Commands and Testing

- `pre-commit run --all-files`: run Ruff lint fixes and formatting.
- `pytest -s -x tests/test_protocol.py`: run a focused CPU test in a prepared environment.
- `pytest -s -x tests/sanity`: check imports and repository health.
- `cd docs && make clean html`: build Sphinx documentation.
- `bash examples/grpo_trainer/run_alfworld.sh`: example **remote-only** training launcher; inspect paths, dependencies, and GPU settings first.

Validate in three levels: (1) local synthetic CPU tests for shapes, masks, ordering, edge cases, NaN/Inf, and hand-checkable outputs; (2) a tiny remote end-to-end smoke test; (3) full experiments. Verify an official PPO/GRPO/GiGPO baseline on the same remote setup before debugging a new method.

## Algorithm and Experiment Rules

Prefer a trajectory-to-credit interface such as `compute_our_advantage(batch, ...)` and reuse existing fields. A “no extra rollout” method must add no branches, environment queries, or hidden trajectories. Keep model, prompts, seeds, rollout budget, optimizer, and schedule fixed between comparisons. Record commit, checkpoint, environment, config, seed, rollout count, max steps, learning rate, batch size, hardware, and framework version.

## Style and Review

Use four-space indentation and Ruff settings from `pyproject.toml`; follow `snake_case`, `CapWords`, and `UPPER_SNAKE_CASE`. Name tests `test_*.py` and `test_*`. Keep commits focused with short imperative subjects. Follow `.github/PULL_REQUEST_TEMPLATE.md`, include validation evidence, document backend impact, update docs, and mark breaking PRs with `[BREAKING]`.
