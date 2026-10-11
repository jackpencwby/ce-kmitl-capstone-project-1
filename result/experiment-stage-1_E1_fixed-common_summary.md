# E1 comparison — shared fixed XGBoost recipe

All five runs use the same source file (SHA-256 `c247a1670b8345b644296e1408bfa7ce2072826f1374582446d2da17d7b7d4a2`), 218 stations, 192,768 rows, and the same split dates: validation starts `2025-11-19`, test starts `2026-04-13`. The row counts are 125,828 train, 31,680 validation, and 35,260 test.

Each run uses the same XGBoost settings at all seven horizons: `n_estimators=2000` maximum, `learning_rate=0.05`, `max_depth=5`, `min_child_weight=20`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0`, `reg_lambda=1`, `gamma=0`, with 100-round validation early stopping. Each model selects its own tree count from validation.

| Experiment | Training strategy | Validation primary macro RMSE ↓ | Test primary macro RMSE ↓ | Run artifacts |
|---|---|---:|---:|---|
| E1_1 | Local | 9.7963 | 5.4830 | [artifact](../experiment-stage-1/artifacts/E1_1_LOCAL__XGB_FIXED_COMMON__DIRECT__NO_NEIGHBOR__SEED42__20261003_133331/) |
| E1_2 | Global | 9.8845 | **4.6373** | [artifact](../experiment-stage-1/artifacts/E1_2_GLOBAL__XGB_FIXED_COMMON__DIRECT__NO_NEIGHBOR__SEED42__20261003_134217/) |
| E1_3 | Global + local tree residual | 10.3026 | 5.0538 | [artifact](../experiment-stage-1/artifacts/E1_3_GLOBAL_XGB_FIXED_COMMON__LOCAL_TREE_RESIDUAL__DIRECT__NO_NEIGHBOR__SEED42__20261003_140236/) |
| E1_4 | Regional | **9.3894** | 4.9038 | [artifact](../experiment-stage-1/artifacts/E1_4_REGIONAL__XGB_FIXED_COMMON__DIRECT__NO_NEIGHBOR__SEED42__20261003_140515/) |
| E1_5 | Global + local MLP residual | 13.3344 | 7.0248 | [artifact](../experiment-stage-1/artifacts/E1_GLOBAL_XGB__LOCAL_MLP_RESIDUAL__DIRECT__NO_NEIGHBOR__SEED42__20261003_143629/) |

Use validation results to choose the training strategy: E1_4 has the lowest validation primary macro RMSE. The test column is the final holdout report and must not be used to choose or retune the strategy.

E1_5 is above the persistence baseline on both splits (10.9931 validation; 5.6568 test), so its local MLP residual correction did not improve this fixed-recipe run.

The E1 scripts use the same fixed train/validation/test boundaries. Their outer walk-forward fold count is zero, as specified by the current E1 holdout policy; E1_3 and E1_5 separately form out-of-fold predictions inside their training window for residual targets. E1_5 additionally uses the fixed MLP settings in `common/config.py`. All runs exported artifacts for horizons 1–7.
