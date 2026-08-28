from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class StudyConfig:
    task: str
    development_file: Path
    external_file: Path
    target: str
    center_column: str
    categorical_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    integer_after_imputation: tuple[str, ...]
    excluded_target_values: tuple[Any, ...]


@dataclass(frozen=True)
class PreprocessingSettings:
    feature_selection: bool
    categorical_missing_value: float
    numerical_imputer_estimators: int
    numerical_imputer_max_depth: int | None
    categorical_imputer_estimators: int
    categorical_imputer_max_depth: int | None
    imputer_nearest_features: int | None
    imputer_max_iter: int
    selector_estimators: int
    selector_criterion: str
    selector_threshold: float | str
    smote_sampling_strategy: float | str
    smote_k_neighbors: int
    enn_sampling_strategy: float | str
    enn_n_neighbors: int
    enn_kind_sel: str


@dataclass
class DataSplit:
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_internal: pd.DataFrame
    y_internal: np.ndarray
    x_external: pd.DataFrame
    y_external: np.ndarray
    train_centers: np.ndarray


@dataclass
class PreparedData:
    x_train_resampled: pd.DataFrame
    y_train_resampled: np.ndarray
    x_train_original: pd.DataFrame
    y_train_original: np.ndarray
    x_internal: pd.DataFrame
    y_internal: np.ndarray
    x_external: pd.DataFrame
    y_external: np.ndarray
    center_train_resampled: dict[Any, pd.DataFrame]
    center_y_resampled: dict[Any, np.ndarray]


def load_study_config(path: str | Path) -> StudyConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    required = {
        "task",
        "development_file",
        "external_file",
        "target",
        "center_column",
        "categorical_columns",
        "numerical_columns",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")

    def resolve_data_path(value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        return candidate.resolve()

    return StudyConfig(
        task=str(raw["task"]),
        development_file=resolve_data_path(raw["development_file"]),
        external_file=resolve_data_path(raw["external_file"]),
        target=str(raw["target"]),
        center_column=str(raw["center_column"]),
        categorical_columns=tuple(raw["categorical_columns"]),
        numerical_columns=tuple(raw["numerical_columns"]),
        integer_after_imputation=tuple(raw.get("integer_after_imputation", ())),
        excluded_target_values=tuple(raw.get("excluded_target_values", ())),
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data format: {path}")


def _validate_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _encode_categorical_pair(
    development: pd.DataFrame,
    external: pd.DataFrame,
    columns: tuple[str, ...],
    missing_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = development.copy()
    external = external.copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(development[column]):
            development[column] = development[column].fillna(missing_value)
            external[column] = pd.to_numeric(
                external[column], errors="coerce"
            ).fillna(missing_value)
            continue

        categories = pd.Index(pd.Series(development[column].dropna().unique()))
        mapping = {value: index for index, value in enumerate(categories)}
        development[column] = development[column].map(mapping).fillna(missing_value)
        external[column] = external[column].map(mapping).fillna(missing_value)
    return development, external


def load_and_split(
    config: StudyConfig,
    train_ratio: float,
    random_state: int,
    categorical_missing_value: float,
) -> DataSplit:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1.")
    development = _read_table(config.development_file)
    external = _read_table(config.external_file)
    feature_columns = [
        column
        for column in config.numerical_columns + config.categorical_columns
        if column != config.center_column
    ]
    required_columns = feature_columns + [config.target, config.center_column]
    _validate_columns(development, required_columns, config.development_file)
    _validate_columns(external, required_columns, config.external_file)

    development = development.loc[development[config.target].notna()].copy()
    external = external.loc[external[config.target].notna()].copy()
    for value in config.excluded_target_values:
        development = development.loc[development[config.target] != value].copy()
        external = external.loc[external[config.target] != value].copy()
    development, external = _encode_categorical_pair(
        development,
        external,
        config.categorical_columns,
        categorical_missing_value,
    )

    train_parts: list[pd.DataFrame] = []
    internal_parts: list[pd.DataFrame] = []
    train_labels: list[np.ndarray] = []
    internal_labels: list[np.ndarray] = []
    train_center_labels: list[np.ndarray] = []

    for center in development[config.center_column].unique():
        center_mask = development[config.center_column] == center
        x_center = development.loc[center_mask, feature_columns]
        y_center = development.loc[center_mask, config.target].astype(float).to_numpy()
        stratify = y_center if len(np.unique(y_center)) > 1 else None
        x_train, x_internal, y_train, y_internal = train_test_split(
            x_center,
            y_center,
            train_size=train_ratio,
            stratify=stratify,
            shuffle=True,
            random_state=random_state,
        )
        train_parts.append(x_train)
        internal_parts.append(x_internal)
        train_labels.append(np.asarray(y_train, dtype=float))
        internal_labels.append(np.asarray(y_internal, dtype=float))
        train_center_labels.append(np.full(len(x_train), center))

    x_train = pd.concat(train_parts, ignore_index=True)
    x_internal = pd.concat(internal_parts, ignore_index=True)
    y_train = np.concatenate(train_labels)
    y_internal = np.concatenate(internal_labels)
    x_external = external.loc[:, feature_columns].reset_index(drop=True)
    y_external = external[config.target].astype(float).to_numpy()

    train_centers = np.concatenate(train_center_labels)
    return DataSplit(
        x_train=x_train,
        y_train=y_train,
        x_internal=x_internal,
        y_internal=y_internal,
        x_external=x_external,
        y_external=y_external,
        train_centers=train_centers,
    )


class ClinicalPreprocessor:
    """Configurable preprocessing fitted only on development data."""

    def __init__(
        self,
        numerical_columns: tuple[str, ...],
        categorical_columns: tuple[str, ...],
        integer_after_imputation: tuple[str, ...],
        settings: PreprocessingSettings,
        random_state: int,
    ) -> None:
        self.numerical_columns = list(numerical_columns)
        self.categorical_columns = list(categorical_columns)
        self.integer_after_imputation = list(integer_after_imputation)
        self.settings = settings
        self.random_state = random_state
        self.imputer: ColumnTransformer | None = None
        self.encoder: ColumnTransformer | None = None
        self.selector: SelectFromModel | None = None
        self.imputed_columns: list[str] = []
        self.encoded_columns: list[str] = []
        self.selected_columns: list[str] = []
        self.resampler = SMOTEENN(
            smote=SMOTE(
                sampling_strategy=settings.smote_sampling_strategy,
                k_neighbors=settings.smote_k_neighbors,
                random_state=random_state,
            ),
            enn=EditedNearestNeighbours(
                sampling_strategy=settings.enn_sampling_strategy,
                n_neighbors=settings.enn_n_neighbors,
                kind_sel=settings.enn_kind_sel,
            ),
            sampling_strategy=settings.enn_sampling_strategy,
            random_state=random_state,
        )

    def fit_transform(self, x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        num_imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=self.settings.numerical_imputer_estimators,
                max_depth=self.settings.numerical_imputer_max_depth,
                random_state=self.random_state,
            ),
            n_nearest_features=self.settings.imputer_nearest_features,
            initial_strategy="mean",
            max_iter=self.settings.imputer_max_iter,
            random_state=self.random_state,
        )
        categorical_imputer = IterativeImputer(
            estimator=RandomForestClassifier(
                n_estimators=self.settings.categorical_imputer_estimators,
                max_depth=self.settings.categorical_imputer_max_depth,
                random_state=self.random_state,
            ),
            initial_strategy="most_frequent",
            missing_values=self.settings.categorical_missing_value,
            max_iter=self.settings.imputer_max_iter,
            random_state=self.random_state,
        )
        self.imputer = ColumnTransformer(
            [
                ("num_impute", num_imputer, self.numerical_columns),
                ("ctg_impute", categorical_imputer, self.categorical_columns),
            ],
            remainder="drop",
        )
        imputed = self.imputer.fit_transform(x)
        self.imputed_columns = [
            name.split("__")[-1] for name in self.imputer.get_feature_names_out()
        ]
        imputed_frame = pd.DataFrame(imputed, columns=self.imputed_columns)
        self._cast_integer_columns(imputed_frame)

        self.encoder = ColumnTransformer(
            [
                ("scale", StandardScaler(), self.numerical_columns),
                (
                    "ctg_encode",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    self.categorical_columns,
                ),
            ],
            remainder="drop",
        )
        encoded = self.encoder.fit_transform(imputed_frame)
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        self.encoded_columns = [
            name.split("__")[-1] for name in self.encoder.get_feature_names_out()
        ]
        encoded_frame = pd.DataFrame(encoded, columns=self.encoded_columns)

        if not self.settings.feature_selection:
            self.selected_columns = self.encoded_columns.copy()
            return encoded_frame

        self.selector = SelectFromModel(
            RandomForestClassifier(
                n_estimators=self.settings.selector_estimators,
                criterion=self.settings.selector_criterion,
                random_state=self.random_state,
            ),
            threshold=self.settings.selector_threshold,
        )
        selected = self.selector.fit_transform(encoded_frame, y)
        self.selected_columns = list(self.selector.get_feature_names_out())
        return pd.DataFrame(selected, columns=self.selected_columns)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.imputer is None or self.encoder is None:
            raise RuntimeError("The preprocessor must be fitted before transform().")
        imputed = self.imputer.transform(x)
        imputed_frame = pd.DataFrame(imputed, columns=self.imputed_columns)
        self._cast_integer_columns(imputed_frame)
        encoded = self.encoder.transform(imputed_frame)
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        encoded_frame = pd.DataFrame(encoded, columns=self.encoded_columns)
        if self.selector is None:
            return encoded_frame
        selected = self.selector.transform(encoded_frame)
        return pd.DataFrame(selected, columns=self.selected_columns)

    def fit_resample(self, x: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        transformed = self.fit_transform(x, y)
        x_resampled, y_resampled = self.resampler.fit_resample(transformed, y)
        return pd.DataFrame(x_resampled, columns=self.selected_columns), np.asarray(
            y_resampled,
            dtype=float,
        )

    def transform_resample(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        transformed = self.transform(x)
        x_resampled, y_resampled = self.resampler.fit_resample(transformed, y)
        return pd.DataFrame(x_resampled, columns=self.selected_columns), np.asarray(
            y_resampled,
            dtype=float,
        )

    def _cast_integer_columns(self, frame: pd.DataFrame) -> None:
        for column in self.integer_after_imputation:
            if column in frame.columns:
                frame[column] = frame[column].astype(int)


def prepare_data(
    split: DataSplit,
    preprocessor: ClinicalPreprocessor,
) -> PreparedData:
    x_train_resampled, y_train_resampled = preprocessor.fit_resample(
        split.x_train,
        split.y_train,
    )
    x_train_original = preprocessor.transform(split.x_train)
    x_internal = preprocessor.transform(split.x_internal)
    x_external = preprocessor.transform(split.x_external)

    center_train_resampled: dict[Any, pd.DataFrame] = {}
    center_y_resampled: dict[Any, np.ndarray] = {}
    for center in np.unique(split.train_centers):
        mask = split.train_centers == center
        center_x, center_y = preprocessor.transform_resample(
            split.x_train.loc[mask].reset_index(drop=True),
            split.y_train[mask],
        )
        center_train_resampled[center] = center_x
        center_y_resampled[center] = center_y

    return PreparedData(
        x_train_resampled=x_train_resampled,
        y_train_resampled=y_train_resampled,
        x_train_original=x_train_original,
        y_train_original=split.y_train,
        x_internal=x_internal,
        y_internal=split.y_internal,
        x_external=x_external,
        y_external=split.y_external,
        center_train_resampled=center_train_resampled,
        center_y_resampled=center_y_resampled,
    )
