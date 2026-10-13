"""Structural regression checks for the four-arm feature-matched ablation."""

import ast
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "scripts" / "run_feature_matched_ablation.py"


def test_user_metric_is_appended_inside_model_loop() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    model_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "model_name"
    ]
    assert model_loops, "The evaluation must iterate over both model arms"

    def is_output_append(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "output"
        )

    assert any(
        any(is_output_append(node) for statement in loop.body for node in ast.walk(statement))
        for loop in model_loops
    ), "Each model arm must append its own user-level result inside the model loop"
