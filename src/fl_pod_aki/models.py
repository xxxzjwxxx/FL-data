from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float
    momentum: float
    weight_decay: float


class MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_layers: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        input_dim = n_features
        for hidden_dim in hidden_layers:
            modules.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
            )
            input_dim = hidden_dim
        modules.extend([nn.Linear(input_dim, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return device


def as_tensor(values: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)


def predict_probability(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x).detach().cpu().numpy().reshape(-1)


def evaluate_auc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    probabilities = predict_probability(model, x)
    return float(roc_auc_score(y.detach().cpu().numpy(), probabilities))


def train_centralized(
    model: MLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_internal: torch.Tensor,
    y_internal: torch.Tensor,
    epochs: int,
    optimization: OptimizationConfig,
    log_every: int,
) -> MLP:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optimization.learning_rate,
        momentum=optimization.momentum,
        weight_decay=optimization.weight_decay,
    )
    criterion = nn.BCELoss(reduction="none")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        probabilities = model(x_train).squeeze()
        loss = criterion(probabilities, y_train).mean()
        loss.backward()
        optimizer.step()
        if log_every and (epoch + 1) % log_every == 0:
            internal_auc = evaluate_auc(model, x_internal, y_internal)
            print(
                f"epoch={epoch + 1} loss={loss.item():.6f} "
                f"internal_auc={internal_auc:.6f}"
            )
    model.eval()
    return model


def train_federated(
    global_model: MLP,
    center_x: dict[Any, torch.Tensor],
    center_y: dict[Any, torch.Tensor],
    x_internal: torch.Tensor,
    y_internal: torch.Tensor,
    algorithm: str,
    rounds: int,
    local_epochs: int,
    optimization: OptimizationConfig,
    fedprox_mu: float | None,
    fedlsd_beta: float | None,
    log_every: int,
) -> MLP:
    if algorithm not in {"fedavg", "fedprox", "fedlsd"}:
        raise ValueError(f"Unsupported federated algorithm: {algorithm}")
    if algorithm == "fedprox" and fedprox_mu is None:
        raise ValueError("fedprox_mu is required for FedProx training.")
    if algorithm == "fedlsd" and fedlsd_beta is None:
        raise ValueError("fedlsd_beta is required for FedLSD training.")

    centers = list(center_x)
    local_models = {
        center: copy.deepcopy(global_model).to(next(global_model.parameters()).device)
        for center in centers
    }
    criterion = nn.BCELoss()

    for round_index in range(rounds):
        global_model.train()
        global_state = copy.deepcopy(global_model.state_dict())
        local_states: list[dict[str, torch.Tensor]] = []

        for center in centers:
            local_model = local_models[center]
            local_model.load_state_dict(global_state)
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=optimization.learning_rate,
                momentum=optimization.momentum,
                weight_decay=optimization.weight_decay,
            )

            for _ in range(local_epochs):
                local_model.train()
                optimizer.zero_grad()
                local_probability = local_model(center_x[center])
                loss = criterion(local_probability.squeeze(), center_y[center])

                if algorithm == "fedprox":
                    assert fedprox_mu is not None
                    proximal = torch.zeros((), device=local_probability.device)
                    for local_parameter, global_parameter in zip(
                        local_model.parameters(),
                        global_model.parameters(),
                    ):
                        proximal = proximal + (fedprox_mu / 2.0) * torch.norm(
                            local_parameter - global_parameter
                        ) ** 2
                    loss = loss + proximal
                elif algorithm == "fedlsd":
                    assert fedlsd_beta is not None
                    global_probability = global_model(center_x[center])
                    distillation = torch.abs(
                        local_probability - global_probability
                    ).sum(-1).mean(0)
                    loss = loss + fedlsd_beta * distillation

                loss.backward()
                optimizer.step()

            local_states.append(copy.deepcopy(local_model.state_dict()))

        center_sizes = np.asarray(
            [len(center_y[center]) for center in centers],
            dtype=float,
        )
        weights = center_sizes / center_sizes.sum()
        aggregated: dict[str, torch.Tensor] = {}
        for key in local_states[0]:
            value = local_states[0][key] * float(weights[0])
            for index in range(1, len(local_states)):
                value = value + local_states[index][key] * float(weights[index])
            aggregated[key] = value
        global_model.load_state_dict(aggregated)

        if log_every and (round_index + 1) % log_every == 0:
            internal_auc = evaluate_auc(global_model, x_internal, y_internal)
            print(
                f"round={round_index + 1} internal_auc={internal_auc:.6f}"
            )

    global_model.eval()
    return global_model
