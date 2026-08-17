import json

from recipe.exact.compare_runs import build_comparison


def _write_run(
    root,
    name,
    algorithm,
    loss_mode,
    lr="1e-6",
    success=0.0,
    commit="a" * 40,
    invalid_penalty="False",
):
    run_dir = root / name
    monitor = run_dir / "monitor"
    monitor.mkdir(parents=True)
    overrides = [
        f"algorithm.adv_estimator={algorithm}",
        f"actor_rollout_ref.actor.loss_agg_mode={loss_mode}",
        f"actor_rollout_ref.actor.use_invalid_action_penalty={invalid_penalty}",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct",
        "env.seed=0",
        "trainer.max_env_steps=60",
        f"actor_rollout_ref.actor.optim.lr={lr}",
        f"trainer.experiment_name={name}",
        f"trainer.default_local_dir={run_dir}/checkpoints",
    ]
    manifest = {
        "experiment": {
            "name": name,
            "environment": "sokoban",
            "algorithm": algorithm,
            "exact_mode": "graph",
            "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
            "seed": 0,
            "loss_agg_mode": loss_mode,
        },
        "git": {"commit": commit},
        "runtime": {
            "python_version": "3.12.0",
            "packages": {"torch": "2.8.0"},
            "gpus": [{"name": "RTX 4090"}],
        },
        "paths": {
            "VERL_AGENT_DATA_ROOT": "/data",
            "TENSORBOARD_DIR": str(run_dir / "tensorboard"),
            "EXACT_CONSOLE_LOG": str(run_dir.with_suffix(".log")),
        },
        "hydra_overrides": overrides,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    (monitor / "heartbeat.json").write_text(json.dumps({"status": "completed", "step": 1}))
    (monitor / "metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "metrics": {
                    "training/global_step": 1,
                    "agent_diag/success_rate": success,
                    "episode/reward/mean": success * 10,
                },
            }
        )
        + "\n"
    )
    return run_dir


def test_compare_runs_accepts_declared_algorithm_and_loss_differences(tmp_path):
    standard = _write_run(
        tmp_path,
        "standard",
        "grpo",
        "token-mean",
        success=0.1,
        invalid_penalty="True",
    )
    matched = _write_run(tmp_path, "matched", "grpo", "seq-mean-token-sum", success=0.2)
    exact = _write_run(tmp_path, "exact", "exact", "seq-mean-token-sum", success=0.3)

    comparison = build_comparison([standard, matched, exact])
    assert comparison["fairness"]["passed"] is True
    assert set(comparison["fairness"]["controlled_override_differences"]) >= {
        "algorithm.adv_estimator",
        "actor_rollout_ref.actor.loss_agg_mode",
        "actor_rollout_ref.actor.use_invalid_action_penalty",
    }
    assert [run["metrics"]["agent_diag/success_rate"]["latest"] for run in comparison["runs"]] == [
        0.1,
        0.2,
        0.3,
    ]


def test_compare_runs_rejects_learning_rate_drift(tmp_path):
    baseline = _write_run(tmp_path, "baseline", "grpo", "seq-mean-token-sum")
    changed = _write_run(tmp_path, "changed", "exact", "seq-mean-token-sum", lr="2e-6")

    comparison = build_comparison([baseline, changed])
    assert comparison["fairness"]["passed"] is False
    assert "actor_rollout_ref.actor.optim.lr" in comparison["fairness"]["uncontrolled_override_differences"]


def test_compare_runs_rejects_commit_drift(tmp_path):
    baseline = _write_run(tmp_path, "baseline", "grpo", "seq-mean-token-sum")
    changed = _write_run(
        tmp_path,
        "changed",
        "exact",
        "seq-mean-token-sum",
        commit="b" * 40,
    )

    comparison = build_comparison([baseline, changed])
    assert comparison["fairness"]["passed"] is False
    assert "git.commit" in comparison["fairness"]["identity_mismatches"]
