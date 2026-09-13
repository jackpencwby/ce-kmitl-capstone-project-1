# PM2.5 forecasting with Darts TFTModel

Open [pm25_tft_training.ipynb](pm25_tft_training.ipynb) using the repository's
CUDA-enabled Python environment. Install `requirements.txt` if necessary.
The configuration cell prints the GPU name and stops if CUDA is unavailable.
The local environment was checked with an RTX 4060 Laptop GPU; training has
not been run for this revision.

Run sections 1-8 to prepare data, train, load the best validation checkpoint,
save artifacts, and evaluate validation. Run section 9 for the final test once
training choices are fixed. Configuration defaults:

| Setting | Default |
|---|---|
| Input / output window | 28 / 7 days |
| Objective | Deterministic MSE; point forecasts |
| Hidden / continuous size | 64 / 16 |
| Dropout / batch size | 0.1 / 64 |
| Maximum epochs | 100, with early stopping patience 12 |
| Learning rate | 0.001, halved on a validation plateau |
| Validation / test | 60 days immediately before the final 60 days |
| Device / precision | CUDA GPU 0 / float32 |

If GPU memory is insufficient, lower `BATCH_SIZE` to 32. Every run saves to a
new `artifacts/r2_<timestamp>/` directory, preserving existing artifacts.
Outputs include the best model and companion checkpoint, `scalers.pkl`,
`run_config.json`, and prediction/group/station/summary CSVs for each split.
Evaluation reports R2, MAE and RMSE for each of days 1-7, plus last-value and
weekly persistence on the same observed labels. Undefined R2 remains missing;
negative R2 is retained. Training uses MSE as a surrogate, so higher held-out
R2 must still be established by your run.

The feature set is unchanged; see [FEATURE_SELECTION.md](FEATURE_SELECTION.md)
for choices and references and [REFERENCE.md](../REFERENCE.md) for TFT details.
Notebook gap filling is causal and scalers see only training dates, but the
upstream CSV's imputation still needs an audit before claiming fully causal
inputs. New splits and observed-only scoring mean old validation scores are
not directly comparable. This model does not produce quantile intervals.

Preparation regression checks (no training):

```powershell
# From repository root
.venv/Scripts/python.exe -m unittest discover -s pm25-tft-model/tests -v
```
