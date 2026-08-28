# Local configuration

Configuration files are intentionally not distributed. They can contain local
data paths, column names, feature schemas, and run settings. Create them only in
your local checkout. All `configs/*.json` files are ignored by Git.

Use two UTF-8 encoded JSON objects:

- a study configuration passed with `--config`;
- a run configuration passed with `--run-config`.

Relative data paths are resolved from the directory containing the study
configuration file.

## Study configuration

| Field | Required | Description |
| --- | --- | --- |
| `task` | Yes | Short label used in local output metadata. |
| `development_file` | Yes | Development-cohort table in CSV, XLS, or XLSX format. |
| `external_file` | Yes | External-validation table in CSV, XLS, or XLSX format. |
| `target` | Yes | Binary outcome column encoded as 0 and 1. |
| `center_column` | Yes | Column identifying the development center. |
| `categorical_columns` | Yes | Array of categorical predictor names. |
| `numerical_columns` | Yes | Array of numerical predictor names. |
| `integer_after_imputation` | No | Numerical columns to round to integers after imputation. |
| `excluded_target_values` | No | Outcome values to exclude before splitting. |

## Run configuration

The run configuration supplies all model, optimization, preprocessing, split,
and federated-learning settings. The public repository deliberately provides no
study-specific values.

Top-level fields:

| Field | Description |
| --- | --- |
| `mode` | Training mode: `central` or `federated`. |
| `algorithm` | Federated aggregation algorithm when federated mode is used. |
| `seed` | Random seed used by splitting, preprocessing, and training. |
| `train_ratio` | Fraction of each development center assigned to training. |
| `hidden_layers` | Array containing the MLP hidden-layer widths. |
| `dropout` | Dropout probability. |
| `decision_threshold` | Probability threshold used for binary predictions. |
| `log_every` | Logging interval. |
| `central_epochs` | Number of epochs used in centralized mode. |
| `rounds` | Number of communication rounds used in federated mode. |
| `local_epochs` | Number of local epochs per federated round. |
| `fedprox_mu` | FedProx proximal coefficient when FedProx is selected. |
| `fedlsd_beta` | FedLSD distillation coefficient when FedLSD is selected. |
| `optimization` | Nested optimization settings described below. |
| `preprocessing` | Nested preprocessing settings described below. |

The `optimization` object contains `learning_rate`, `momentum`, and
`weight_decay`.

The `preprocessing` object contains:

- `feature_selection`
- `categorical_missing_value`
- `numerical_imputer_estimators`
- `numerical_imputer_max_depth`
- `categorical_imputer_estimators`
- `categorical_imputer_max_depth`
- `imputer_nearest_features`
- `imputer_max_iter`
- `selector_estimators`
- `selector_criterion`
- `selector_threshold`
- `smote_sampling_strategy`
- `smote_k_neighbors`
- `enn_sampling_strategy`
- `enn_n_neighbors`
- `enn_kind_sel`

Both cohort tables must contain the configured target, center, and predictor
columns. Keep clinical files and local configuration files outside version
control.

Run after creating both private files:

```bash
python -m fl_pod_aki.train --config configs/study.local.json --run-config configs/run.local.json --output-dir outputs/run
```
