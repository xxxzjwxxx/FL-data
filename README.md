# FL-POD-AKI

Source code for centralized and federated multilayer perceptron models used to
predict postoperative delirium (POD) and acute kidney injury (AKI).

## Publication

This repository accompanies the following article:

> Qian Wang, Yu-Xiang Song, Xiao-Dong Yang, Jing-Wei Zhang, et al.
> **[Multicenter Privacy-Preserving Federated Models for Predicting Postoperative
> Delirium and Acute Kidney Injury in Older
> Patients](https://doi.org/10.1002/mco2.70944).**
> *[MedComm](https://onlinelibrary.wiley.com/journal/26882663)*.
> 2026;7(9):e70944. PubMed: [42633270](https://pubmed.ncbi.nlm.nih.gov/42633270/).

## What is included

- Configurable center-stratified development/internal-validation splitting.
- Development-only preprocessing with random-forest imputation, scaling,
  one-hot encoding, optional feature selection, and optional resampling.
- Centralized MLP, FedAvg, FedLSD, and FedProx training.
- Local export of predictions, metrics, model artifacts, and run metadata.

Study data and all locally generated artifacts are not included. Runtime
settings are supplied through untracked local configuration files.

## Repository layout

```text
configs/             Instructions for creating private local configuration
data/                Local private data location; contents are gitignored
src/fl_pod_aki/      Data preparation and centralized/federated model training
tests/               Synthetic tests without clinical data
```

## Environment

The package targets Python 3.10. Create an isolated environment and install the
pinned software dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\activate`. For GPU training, install a
PyTorch 2.1.1 build compatible with the local CUDA runtime before installing the
project.

## Private data

Place authorized, de-identified files at `data/development.xlsx` and
`data/external.xlsx`. Both must contain the target, center, and predictor columns
declared in the private study configuration. Data files are ignored by Git.

## Run

Create separate local JSON files for the data schema and runtime settings by
following `configs/README.md`. No concrete study settings are distributed.

```bash
python -m fl_pod_aki.train --config configs/study.local.json --run-config configs/run.local.json --output-dir outputs/run
```

Both local JSON files and all generated outputs are ignored by Git.

## Tests and release check

```bash
python -m pip install -e ".[test]"
pytest
python tools/audit_release.py
```

The audit rejects private paths, institution identifiers, IP addresses,
spreadsheet data, model artifacts, and generated outputs in the tracked release
tree.

## Data and licensing

Data access remains subject to the study approvals and the participating data
custodians. No clinical data are redistributed here. A repository license is
not assigned by this technical release; the copyright holder should select and
add the intended license before making the GitHub repository public.
