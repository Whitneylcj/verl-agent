import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_SOURCE = REPO_ROOT / "verl" / "workers" / "fsdp_workers.py"


def _worker_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(WORKER_SOURCE.read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    return next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_init_builds_rollout_before_sharding_manager_is_used() -> None:
    init_model = _worker_method("init_model")
    build_rollout_guard = next(node for node in ast.walk(init_model) if isinstance(node, ast.If) and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "_build_rollout" for call in ast.walk(node)))

    assert ast.unparse(build_rollout_guard.test) == "self._is_rollout"


def test_update_stages_rollout_weights_before_actor_offload() -> None:
    update_actor = _worker_method("update_actor")
    calls = [node for node in ast.walk(update_actor) if isinstance(node, ast.Call)]
    stage = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "stage_updated_weights")
    offload = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "offload_fsdp_model_to_cpu")
    synchronize = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "synchronize")
    final_empty_cache = max(node.lineno for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "empty_cache")

    assert stage.lineno < offload.lineno < synchronize.lineno < final_empty_cache
