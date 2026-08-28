from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import (
    ClinicalPreprocessor,
    PreprocessingSettings,
    load_and_split,
    load_study_config,
    prepare_data,
)
from .models import (
    MLP,
    OptimizationConfig,
    as_tensor,
    predict_probability,
    resolve_device,
    set_reproducible_seed,
    train_centralized,
    train_federated,
)


@dataclass(frozen=True)
class TrainingConfig:
    mode: str
    algorithm: str | None
    seed: int
    train_ratio: float
    hidden_layers: tuple[int, ...]
    dropout: float
    learning_rate: float
    momentum: float
    weight_decay: float
    decision_threshold: float
    central_epochs: int | None
    rounds: int | None
    local_epochs: int | None
    fedprox_mu: float | None
    fedlsd_beta: float | None
    log_every: int
    preprocessing: PreprocessingSettings


def _require_keys(values: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys.difference(values))
    if missing:
        raise ValueError(f"Missing {label} keys: {', '.join(missing)}")


def _positive_int(value: Any, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive.")
    return parsed


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


def _strategy(value: Any) -> float | str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return str(value)


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    _require_keys(
        raw,
        {
            "mode",
            "seed",
            "train_ratio",
            "hidden_layers",
            "dropout",
            "decision_threshold",
            "log_every",
            "optimization",
            "preprocessing",
        },
        "training configuration",
    )
    optimization = raw["optimization"]
    preprocessing = raw["preprocessing"]
    _require_keys(
        optimization,
        {"learning_rate", "momentum", "weight_decay"},
        "optimization",
    )
    _require_keys(
        preprocessing,
        {
            "feature_selection",
            "categorical_missing_value",
            "numerical_imputer_estimators",
            "numerical_imputer_max_depth",
            "categorical_imputer_estimators",
            "categorical_imputer_max_depth",
            "imputer_nearest_features",
            "imputer_max_iter",
            "selector_estimators",
            "selector_criterion",
            "selector_threshold",
            "smote_sampling_strategy",
            "smote_k_neighbors",
            "enn_sampling_strategy",
            "enn_n_neighbors",
            "enn_kind_sel",
        },
        "preprocessing",
    )

    mode = str(raw["mode"]).lower()
    if mode not in {"central", "federated"}:
        raise ValueError("mode must be 'central' or 'federated'.")
    algorithm: str | None = None
    central_epochs: int | None = None
    rounds: int | None = None
    local_epochs: int | None = None
    fedprox_mu: float | None = None
    fedlsd_beta: float | None = None
    if mode == "central":
        _require_keys(raw, {"central_epochs"}, "central training")
        central_epochs = _positive_int(raw["central_epochs"], "central_epochs")
    else:
        _require_keys(
            raw,
            {"algorithm", "rounds", "local_epochs"},
            "federated training",
        )
        algorithm = str(raw["algorithm"]).lower()
        if algorithm not in {"fedavg", "fedlsd", "fedprox"}:
            raise ValueError("Unsupported federated algorithm.")
        rounds = _positive_int(raw["rounds"], "rounds")
        local_epochs = _positive_int(raw["local_epochs"], "local_epochs")
        if algorithm == "fedprox":
            _require_keys(raw, {"fedprox_mu"}, "FedProx")
            fedprox_mu = float(raw["fedprox_mu"])
        elif algorithm == "fedlsd":
            _require_keys(raw, {"fedlsd_beta"}, "FedLSD")
            fedlsd_beta = float(raw["fedlsd_beta"])

    hidden_layers = tuple(int(value) for value in raw["hidden_layers"])
    if not hidden_layers or any(value <= 0 for value in hidden_layers):
        raise ValueError("hidden_layers must contain positive integers.")
    train_ratio = float(raw["train_ratio"])
    dropout = float(raw["dropout"])
    decision_threshold = float(raw["decision_threshold"])
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be in [0, 1].")
    if not isinstance(preprocessing["feature_selection"], bool):
        raise ValueError("preprocessing.feature_selection must be boolean.")

    return TrainingConfig(
        mode=mode,
        algorithm=algorithm,
        seed=int(raw["seed"]),
        train_ratio=train_ratio,
        hidden_layers=hidden_layers,
        dropout=dropout,
        learning_rate=float(optimization["learning_rate"]),
        momentum=float(optimization["momentum"]),
        weight_decay=float(optimization["weight_decay"]),
        decision_threshold=decision_threshold,
        central_epochs=central_epochs,
        rounds=rounds,
        local_epochs=local_epochs,
        fedprox_mu=fedprox_mu,
        fedlsd_beta=fedlsd_beta,
        log_every=max(0, int(raw["log_every"])),
        preprocessing=PreprocessingSettings(
            feature_selection=preprocessing["feature_selection"],
            categorical_missing_value=float(
                preprocessing["categorical_missing_value"]
            ),
            numerical_imputer_estimators=_positive_int(
                preprocessing["numerical_imputer_estimators"],
                "numerical_imputer_estimators",
            ),
            numerical_imputer_max_depth=_optional_positive_int(
                preprocessing["numerical_imputer_max_depth"],
                "numerical_imputer_max_depth",
            ),
            categorical_imputer_estimators=_positive_int(
                preprocessing["categorical_imputer_estimators"],
                "categorical_imputer_estimators",
            ),
            categorical_imputer_max_depth=_optional_positive_int(
                preprocessing["categorical_imputer_max_depth"],
                "categorical_imputer_max_depth",
            ),
            imputer_nearest_features=_optional_positive_int(
                preprocessing["imputer_nearest_features"],
                "imputer_nearest_features",
            ),
            imputer_max_iter=_positive_int(
                preprocessing["imputer_max_iter"], "imputer_max_iter"
            ),
            selector_estimators=_positive_int(
                preprocessing["selector_estimators"], "selector_estimators"
            ),
            selector_criterion=str(preprocessing["selector_criterion"]),
            selector_threshold=_strategy(preprocessing["selector_threshold"]),
            smote_sampling_strategy=_strategy(
                preprocessing["smote_sampling_strategy"]
            ),
            smote_k_neighbors=_positive_int(
                preprocessing["smote_k_neighbors"], "smote_k_neighbors"
            ),
            enn_sampling_strategy=_strategy(preprocessing["enn_sampling_strategy"]),
            enn_n_neighbors=_positive_int(
                preprocessing["enn_n_neighbors"], "enn_n_neighbors"
            ),
            enn_kind_sel=str(preprocessing["enn_kind_sel"]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a configurable centralized or federated MLP model."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-artifacts", action="store_true")
    return parser.parse_args()


def default_output_dir(task: str, model_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / task.lower() / f"{timestamp}_{model_stem}"


def save_predictions(
    path: Path,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        {
            "True_Label": np.asarray(labels, dtype=float).reshape(-1),
            "Prob_Label": np.asarray(probabilities, dtype=float).reshape(-1),
        }
    )
    frame.to_excel(path, index=False)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "N": len(labels),
        "Events": int(labels.sum()),
        "Event rate": float(labels.mean()),
        "Mean predicted": float(probabilities.mean()),
        "AUC": float(roc_auc_score(labels, probabilities)),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "Specificity": float(specificity),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "Threshold": float(threshold),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    training = load_training_config(args.run_config)

    set_reproducible_seed(training.seed)
    device = resolve_device(args.device)
    split = load_and_split(
        config,
        train_ratio=training.train_ratio,
        random_state=training.seed,
        categorical_missing_value=training.preprocessing.categorical_missing_value,
    )
    numerical_columns = tuple(
        column for column in config.numerical_columns if column != config.center_column
    )
    categorical_columns = tuple(
        column for column in config.categorical_columns if column != config.center_column
    )
    preprocessor = ClinicalPreprocessor(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        integer_after_imputation=config.integer_after_imputation,
        settings=training.preprocessing,
        random_state=training.seed,
    )
    prepared = prepare_data(split, preprocessor)

    x_train_resampled = as_tensor(prepared.x_train_resampled, device)
    y_train_resampled = as_tensor(prepared.y_train_resampled, device)
    x_internal = as_tensor(prepared.x_internal, device)
    y_internal = as_tensor(prepared.y_internal, device)
    x_external = as_tensor(prepared.x_external, device)
    x_train_original = as_tensor(prepared.x_train_original, device)

    model = MLP(
        n_features=x_train_resampled.shape[1],
        hidden_layers=training.hidden_layers,
        dropout=training.dropout,
    ).to(device)
    optimization = OptimizationConfig(
        learning_rate=training.learning_rate,
        momentum=training.momentum,
        weight_decay=training.weight_decay,
    )

    if training.mode == "central":
        assert training.central_epochs is not None
        model_stem = "centralized"
        model = train_centralized(
            model,
            x_train_resampled,
            y_train_resampled,
            x_internal,
            y_internal,
            epochs=training.central_epochs,
            optimization=optimization,
            log_every=training.log_every,
        )
    else:
        assert training.algorithm is not None
        assert training.rounds is not None
        assert training.local_epochs is not None
        model_stem = training.algorithm
        center_x = {
            center: as_tensor(values, device)
            for center, values in prepared.center_train_resampled.items()
        }
        center_y = {
            center: as_tensor(values, device)
            for center, values in prepared.center_y_resampled.items()
        }
        model = train_federated(
            model,
            center_x,
            center_y,
            x_internal,
            y_internal,
            algorithm=training.algorithm,
            rounds=training.rounds,
            local_epochs=training.local_epochs,
            optimization=optimization,
            fedprox_mu=training.fedprox_mu,
            fedlsd_beta=training.fedlsd_beta,
            log_every=training.log_every,
        )

    output_dir = (args.output_dir or default_output_dir(config.task, model_stem)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_sets = {
        "development_original": (
            prepared.y_train_original,
            predict_probability(model, x_train_original),
            output_dir / "development_predictions.xlsx",
        ),
        "internal": (
            prepared.y_internal,
            predict_probability(model, x_internal),
            output_dir / "internal_predictions.xlsx",
        ),
        "external": (
            prepared.y_external,
            predict_probability(model, x_external),
            output_dir / "external_predictions.xlsx",
        ),
    }

    metric_rows: list[dict[str, Any]] = []
    for dataset, (labels, probabilities, path) in prediction_sets.items():
        save_predictions(path, labels, probabilities)
        metric_rows.append(
            {
                "Dataset": dataset,
                **classification_metrics(
                    labels,
                    probabilities,
                    training.decision_threshold,
                ),
            }
        )
    pd.DataFrame(metric_rows).to_csv(output_dir / "classification_metrics.csv", index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": config.task,
        "training": asdict(training),
        "device": str(device),
        "study_config_sha256": sha256(Path(args.config).expanduser().resolve()),
        "run_config_sha256": sha256(Path(args.run_config).expanduser().resolve()),
        "runtime": runtime_versions(),
        "prediction_files": {
            dataset: path.name for dataset, (_, _, path) in prediction_sets.items()
        },
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    if args.save_artifacts:
        torch.save(model.state_dict(), output_dir / "model_state.pt")
        joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
        with (output_dir / "model_structure.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "n_features": int(x_train_resampled.shape[1]),
                    "hidden_layers": training.hidden_layers,
                    "dropout": training.dropout,
                    "selected_features": preprocessor.selected_columns,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"Saved run to: {output_dir}")


if __name__ == "__main__":
    main()
