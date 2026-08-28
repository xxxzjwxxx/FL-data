# Data placement

The clinical records used in the study are not included in this repository.
Place locally authorized, de-identified files here as:

```text
data/
  development.xlsx
  external.xlsx
```

Both files must contain the configured feature columns, a `center` column, and
the outcome columns. Define the corresponding column names in a private local
JSON configuration as described in `configs/README.md`.

Categorical columns should be integer-coded consistently across development and
external files. This avoids category-order ambiguity across institutions.

The development file contains the multicenter development cohort. The code
performs a center-stratified split using the ratio and random seed supplied in
the private run configuration. The external file is used only for external
evaluation and is never used to fit the model, preprocessor, or feature
selector.

Do not commit data, predictions, fitted models, or preprocessing artifacts.
The repository `.gitignore` excludes these files by default.
