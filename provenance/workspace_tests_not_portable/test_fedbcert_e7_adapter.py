from __future__ import annotations

from pathlib import Path

from scripts.run_boundary_certified_external_baseline import (
    build_parser,
    framework_args,
)


def parsed_args():
    return build_parser().parse_args(
        [
            "--baseline",
            "feddep",
            "--dataset",
            "Cora",
            "--partition",
            "metis",
            "--seed",
            "11",
        ]
    )


def test_e7_adapter_exposes_task_optimizer() -> None:
    args = parsed_args()
    adapted = framework_args(args)
    assert adapted.optim == "adam"
    assert adapted.lr == args.learning_rate
    assert adapted.local_lr == args.learning_rate
    assert adapted.weight_decay == args.weight_decay


def test_e7_adapter_contains_required_node_task_fields() -> None:
    adapted = framework_args(parsed_args())
    required = {
        "task",
        "dataset",
        "num_clients",
        "train_val_test",
        "processing",
        "processing_percentage",
        "model",
        "hid_dim",
        "num_layers",
        "dropout",
        "weight_decay",
        "optim",
        "local_lr",
        "lr",
        "num_epochs",
        "dp_mech",
        "grad_clip",
        "metrics",
        "fl_algorithm",
        "batch_size",
    }
    assert not (required - vars(adapted).keys())


def test_feddep_custom_evaluator_returns_logits() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "openfgl/flcore/feddep/client.py",
        "openfgl/flcore/feddep/server.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert 'eval_output["logits"] = logits' in source


def test_e7_evaluation_aligns_masks_and_labels_with_logits() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/run_boundary_certified_external_baseline.py"
    ).read_text(encoding="utf-8")
    assert ".bool().to(logits.device)" in source
    assert "client.task.data.y[:original_n].to(logits.device)" in source
