from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fl_pod_aki.models import (
    MLP,
    OptimizationConfig,
    predict_probability,
    set_reproducible_seed,
    train_centralized,
    train_federated,
)


def _binary_data() -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [[-1.0, -0.5], [-0.5, -1.0], [0.5, 1.0], [1.0, 0.5]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    return features, labels


def test_centralized_training_produces_probabilities() -> None:
    set_reproducible_seed(7)
    features, labels = _binary_data()
    model = MLP(n_features=2, hidden_layers=(4,), dropout=0.0)
    trained = train_centralized(
        model,
        features,
        labels,
        features,
        labels,
        epochs=2,
        optimization=OptimizationConfig(0.01, 0.0, 0.0),
        log_every=0,
    )
    probabilities = predict_probability(trained, features)
    assert probabilities.shape == (4,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


@pytest.mark.parametrize("algorithm", ["fedavg", "fedlsd", "fedprox"])
def test_federated_algorithms_complete_one_round(algorithm: str) -> None:
    set_reproducible_seed(7)
    features, labels = _binary_data()
    center_x = {"a": features.clone(), "b": features.flip(0).clone()}
    center_y = {"a": labels.clone(), "b": labels.flip(0).clone()}
    model = MLP(n_features=2, hidden_layers=(4,), dropout=0.0)
    trained = train_federated(
        model,
        center_x,
        center_y,
        features,
        labels,
        algorithm=algorithm,
        rounds=1,
        local_epochs=1,
        optimization=OptimizationConfig(0.01, 0.0, 0.0),
        fedprox_mu=0.2,
        fedlsd_beta=0.3,
        log_every=0,
    )
    probabilities = predict_probability(trained, features)
    assert np.isfinite(probabilities).all()
