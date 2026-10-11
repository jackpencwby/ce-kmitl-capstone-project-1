# E1-E4 Experiment Comparison by Metric

Validation and Test values are shown side by side. The `[TEST BEST]` label is descriptive only. Choose configurations using the primary validation metric; never select or retune from Test.
Lower is better for error metrics. Higher is better for R2. For bias, compare absolute distance from zero.

## Reading guide

- Primary model-selection metric: `primary_macro_rmse` (mean RMSE across station x horizon groups).
- Micro metrics pool all prediction rows. Macro metrics average across stations or station x horizon groups.
- `[TEST BEST]` identifies the best observed Test value within that E group for reference; Validation controls model selection.
- E1.1-E1.5 were rerun with one shared fixed XGBoost recipe on the same master dataset hash (`c247a167...`), stations, and split dates. E1.5 adds its fixed local MLP residual corrector.
- E3 has 215,257 test pairs; E1, E2, and E4 generally have 229,588. Compare winners within each E group, not across groups.
- `N/A` means unavailable.

## Dataset and Evaluation Support

| Experiment | Method | Master dataset SHA-256 prefix | Validation n | Test n |
|---|---|---|---:|---:|
| E1_1 | Local XGBoost | `c247a167...` | 205,287 | 229,588 |
| E1_2 | Global XGBoost | `c247a167...` | 205,287 | 229,588 |
| E1_3 | Global + local tree residual | `c247a167...` | 205,287 | 229,588 |
| E1_4 | Regional XGBoost | `c247a167...` | 205,287 | 229,588 |
| E1_5 | Global + local MLP residual | `c247a167...` | 205,287 | 229,588 |
| E2_1 | XGBoost | `c247a167...` | 205,287 | 229,588 |
| E2_2 | LightGBM | `c247a167...` | 205,287 | 229,588 |
| E2_3 | GradientBoostingRegressor | `c247a167...` | 205,287 | 229,588 |
| E3_1 | Direct forecast | `c247a167...` | 199,339 | 215,257 |
| E3_2 | Multi-output forecast | `c247a167...` | 199,339 | 215,257 |
| E4_1 | No neighbor features | `7e160e7b...` | 205,287 | 229,588 |
| E4_2 | Unweighted neighbor features | `7e160e7b...` | 205,287 | 229,588 |
| E4_3 | Distance-weighted neighbors | `7e160e7b...` | 205,287 | 229,588 |
| E4_4 | Wind + distance weighted neighbors | `7e160e7b...` | 205,287 | 229,588 |

E1 fixed-common parameters: `learning_rate=0.05`, `max_depth=5`, `min_child_weight=20`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0`, `reg_lambda=1`, `gamma=0`, and `n_estimators=2000` maximum with 100-round validation early stopping. All E1 runs use 218 stations and split starts `2025-11-19` (validation) and `2026-04-13` (test).

## Micro Metrics

### Micro MAE (ug/m3)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 7.4218 | 4.1504 |
| E1 | E1_2 | Global XGBoost | 7.5546 | **3.2492 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 7.7167 | 3.4892 |
| E1 | E1_4 | Regional XGBoost | 7.1366 | 3.5661 |
| E1 | E1_5 | Global + local MLP residual | 9.1330 | 3.9806 |
| E2 | E2_1 | XGBoost | 7.7058 | **3.2803 [TEST BEST]** |
| E2 | E2_2 | LightGBM | 7.7248 | 3.3007 |
| E2 | E2_3 | GradientBoostingRegressor | 8.0774 | 3.5276 |
| E3 | E3_1 | Direct forecast | 7.9998 | **4.5145 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 7.9943 | 4.5670 |
| E4 | E4_1 | No neighbor features | 7.9811 | 4.4675 |
| E4 | E4_2 | Unweighted neighbor features | 7.9804 | 4.4702 |
| E4 | E4_3 | Distance-weighted neighbors | 7.9808 | **4.4550 [TEST BEST]** |
| E4 | E4_4 | Wind + distance weighted neighbors | 7.9878 | 4.4703 |

### Micro MSE ((ug/m3)^2)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 116.7430 | 35.8440 |
| E1 | E1_2 | Global XGBoost | 124.4762 | **25.3223 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 139.8952 | 31.1134 |
| E1 | E1_4 | Regional XGBoost | 113.1674 | 27.7694 |
| E1 | E1_5 | Global + local MLP residual | 602.4313 | 179.7775 |
| E2 | E2_1 | XGBoost | 127.6012 | 25.2149 |
| E2 | E2_2 | LightGBM | 127.7928 | **24.4081 [TEST BEST]** |
| E2 | E2_3 | GradientBoostingRegressor | 140.1546 | 28.4370 |
| E3 | E3_1 | Direct forecast | 132.9113 | **39.3429 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 133.2260 | 40.1098 |
| E4 | E4_1 | No neighbor features | 133.2267 | **38.2712 [TEST BEST]** |
| E4 | E4_2 | Unweighted neighbor features | 133.2646 | 38.6334 |
| E4 | E4_3 | Distance-weighted neighbors | 133.3667 | 38.3012 |
| E4 | E4_4 | Wind + distance weighted neighbors | 133.9595 | 38.4225 |

### Micro RMSE (ug/m3)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 10.8048 | 5.9870 |
| E1 | E1_2 | Global XGBoost | 11.1569 | **5.0321 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 11.8277 | 5.5779 |
| E1 | E1_4 | Regional XGBoost | 10.6380 | 5.2697 |
| E1 | E1_5 | Global + local MLP residual | 24.5445 | 13.4081 |
| E2 | E2_1 | XGBoost | 11.2961 | 5.0214 |
| E2 | E2_2 | LightGBM | 11.3045 | **4.9405 [TEST BEST]** |
| E2 | E2_3 | GradientBoostingRegressor | 11.8387 | 5.3326 |
| E3 | E3_1 | Direct forecast | 11.5287 | **6.2724 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 11.5424 | 6.3332 |
| E4 | E4_1 | No neighbor features | 11.5424 | **6.1864 [TEST BEST]** |
| E4 | E4_2 | Unweighted neighbor features | 11.5440 | 6.2156 |
| E4 | E4_3 | Distance-weighted neighbors | 11.5484 | 6.1888 |
| E4 | E4_4 | Wind + distance weighted neighbors | 11.5741 | 6.1986 |

### Micro R2 (score)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 0.5949 | 0.6334 |
| E1 | E1_2 | Global XGBoost | 0.5681 | **0.7410 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 0.5146 | 0.6818 |
| E1 | E1_4 | Regional XGBoost | 0.6073 | 0.7160 |
| E1 | E1_5 | Global + local MLP residual | -1.0903 | -0.8385 |
| E2 | E2_1 | XGBoost | 0.5573 | 0.7421 |
| E2 | E2_2 | LightGBM | 0.5566 | **0.7504 [TEST BEST]** |
| E2 | E2_3 | GradientBoostingRegressor | 0.5137 | 0.7092 |
| E3 | E3_1 | Direct forecast | 0.5153 | **0.6174 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 0.5141 | 0.6099 |
| E4 | E4_1 | No neighbor features | 0.5377 | **0.6086 [TEST BEST]** |
| E4 | E4_2 | Unweighted neighbor features | 0.5376 | 0.6049 |
| E4 | E4_3 | Distance-weighted neighbors | 0.5373 | 0.6083 |
| E4 | E4_4 | Wind + distance weighted neighbors | 0.5352 | 0.6071 |

### Micro bias (ug/m3; closest to zero)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | -1.5571 | 1.7756 |
| E1 | E1_2 | Global XGBoost | 0.0362 | 0.3996 |
| E1 | E1_3 | Global + local tree residual | -1.4557 | -1.2393 |
| E1 | E1_4 | Regional XGBoost | -0.3121 | 0.9891 |
| E1 | E1_5 | Global + local MLP residual | -2.3804 | **-1.5006 [TEST BEST]** |
| E2 | E2_1 | XGBoost | 0.0775 | **0.7217 [TEST BEST]** |
| E2 | E2_2 | LightGBM | 0.1041 | 0.8120 |
| E2 | E2_3 | GradientBoostingRegressor | -0.0077 | 0.7897 |
| E3 | E3_1 | Direct forecast | -0.3329 | **2.5582 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | -0.2002 | 2.5955 |
| E4 | E4_1 | No neighbor features | -0.4540 | 2.5324 |
| E4 | E4_2 | Unweighted neighbor features | -0.4758 | 2.5303 |
| E4 | E4_3 | Distance-weighted neighbors | -0.4700 | **2.5280 [TEST BEST]** |
| E4 | E4_4 | Wind + distance weighted neighbors | -0.4826 | 2.5384 |

### High-PM MAE (true PM2.5 > 37.5) (ug/m3)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 13.1992 | 12.8728 |
| E1 | E1_2 | Global XGBoost | 13.7148 | **12.1475 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 15.1743 | 15.2220 |
| E1 | E1_4 | Regional XGBoost | 13.1099 | 12.8354 |
| E1 | E1_5 | Global + local MLP residual | 17.2759 | 20.9668 |
| E2 | E2_1 | XGBoost | 13.7594 | 10.4416 |
| E2 | E2_2 | LightGBM | 13.7233 | **10.1007 [TEST BEST]** |
| E2 | E2_3 | GradientBoostingRegressor | 14.0154 | 10.9934 |
| E3 | E3_1 | Direct forecast | 12.9198 | **11.9613 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 12.8792 | 12.4072 |
| E4 | E4_1 | No neighbor features | 12.9824 | **11.9844 [TEST BEST]** |
| E4 | E4_2 | Unweighted neighbor features | 12.9745 | 12.1042 |
| E4 | E4_3 | Distance-weighted neighbors | 12.9737 | 12.0487 |
| E4 | E4_4 | Wind + distance weighted neighbors | 13.0136 | 12.0853 |

### High-PM RMSE (true PM2.5 > 37.5) (ug/m3)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 18.1156 | 17.4448 |
| E1 | E1_2 | Global XGBoost | 19.0951 | **16.6043 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 21.2653 | 20.9313 |
| E1 | E1_4 | Regional XGBoost | 17.8983 | 17.3798 |
| E1 | E1_5 | Global + local MLP residual | 31.8416 | 40.4207 |
| E2 | E2_1 | XGBoost | 18.9490 | 13.8500 |
| E2 | E2_2 | LightGBM | 18.8411 | **13.4667 [TEST BEST]** |
| E2 | E2_3 | GradientBoostingRegressor | 19.2518 | 14.5724 |
| E3 | E3_1 | Direct forecast | 18.5068 | **16.0011 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 18.5185 | 16.6339 |
| E4 | E4_1 | No neighbor features | 18.5453 | **16.0279 [TEST BEST]** |
| E4 | E4_2 | Unweighted neighbor features | 18.5542 | 16.2185 |
| E4 | E4_3 | Distance-weighted neighbors | 18.5629 | 16.1161 |
| E4 | E4_4 | Wind + distance weighted neighbors | 18.6465 | 16.1613 |

### High-PM MAE (true PM2.5 > 75) (ug/m3)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 32.1930 | **21.4841 [TEST BEST]** |
| E1 | E1_2 | Global XGBoost | 36.1002 | 24.5485 |
| E1 | E1_3 | Global + local tree residual | 41.7750 | 31.9558 |
| E1 | E1_4 | Regional XGBoost | 32.3386 | 24.8298 |
| E1 | E1_5 | Global + local MLP residual | 48.0555 | 41.5055 |
| E2 | E2_1 | XGBoost | 35.3705 | **14.6590 [TEST BEST]** |
| E2 | E2_2 | LightGBM | 34.9059 | 15.2004 |
| E2 | E2_3 | GradientBoostingRegressor | 35.6118 | 17.2089 |
| E3 | E3_1 | Direct forecast | 37.2585 | **17.1333 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 37.3671 | 18.8772 |
| E4 | E4_1 | No neighbor features | 35.1115 | 17.1820 |
| E4 | E4_2 | Unweighted neighbor features | 35.1959 | 17.3657 |
| E4 | E4_3 | Distance-weighted neighbors | 35.2329 | **17.1334 [TEST BEST]** |
| E4 | E4_4 | Wind + distance weighted neighbors | 35.4188 | 17.1725 |

### High-PM RMSE (true PM2.5 > 75) (ug/m3)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 38.7581 | **27.5291 [TEST BEST]** |
| E1 | E1_2 | Global XGBoost | 42.6816 | 29.0998 |
| E1 | E1_3 | Global + local tree residual | 48.6131 | 37.1709 |
| E1 | E1_4 | Regional XGBoost | 38.4636 | 29.7167 |
| E1 | E1_5 | Global + local MLP residual | 60.7925 | 63.2027 |
| E2 | E2_1 | XGBoost | 41.7500 | **18.8754 [TEST BEST]** |
| E2 | E2_2 | LightGBM | 41.2467 | 19.7009 |
| E2 | E2_3 | GradientBoostingRegressor | 42.0033 | 21.1859 |
| E3 | E3_1 | Direct forecast | 43.9502 | **22.2552 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 44.1287 | 24.3326 |
| E4 | E4_1 | No neighbor features | 41.8893 | 22.3890 |
| E4 | E4_2 | Unweighted neighbor features | 41.9632 | 22.6858 |
| E4 | E4_3 | Distance-weighted neighbors | 42.0134 | **22.3632 [TEST BEST]** |
| E4 | E4_4 | Wind + distance weighted neighbors | 42.2507 | 22.4997 |

## Macro Metrics

### Macro RMSE by station (ug/m3)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 9.9464 | 5.5511 |
| E1 | E1_2 | Global XGBoost | 10.1032 | **4.6849 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 10.5479 | 5.1192 |
| E1 | E1_4 | Regional XGBoost | 9.5810 | 4.9609 |
| E1 | E1_5 | Global + local MLP residual | 13.9383 | 7.4858 |
| E2 | E2_1 | XGBoost | 10.2977 | 4.6890 |
| E2 | E2_2 | LightGBM | 10.3143 | **4.6438 [TEST BEST]** |
| E2 | E2_3 | GradientBoostingRegressor | 10.9099 | 4.9578 |
| E3 | E3_1 | Direct forecast | 10.6563 | **5.9044 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 10.6528 | 5.9422 |
| E4 | E4_1 | No neighbor features | 10.6461 | 5.8352 |
| E4 | E4_2 | Unweighted neighbor features | 10.6460 | 5.8509 |
| E4 | E4_3 | Distance-weighted neighbors | 10.6492 | **5.8333 [TEST BEST]** |
| E4 | E4_4 | Wind + distance weighted neighbors | 10.6653 | 5.8414 |

### Macro MAE by station (ug/m3)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 7.3902 | 4.1507 |
| E1 | E1_2 | Global XGBoost | 7.4952 | **3.2490 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 7.6543 | 3.4909 |
| E1 | E1_4 | Regional XGBoost | 7.0736 | 3.5637 |
| E1 | E1_5 | Global + local MLP residual | 9.1792 | 4.0156 |
| E2 | E2_1 | XGBoost | 7.6455 | **3.2804 [TEST BEST]** |
| E2 | E2_2 | LightGBM | 7.6637 | 3.3004 |
| E2 | E2_3 | GradientBoostingRegressor | 8.0157 | 3.5292 |
| E3 | E3_1 | Direct forecast | 7.9623 | **4.5143 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 7.9559 | 4.5663 |
| E4 | E4_1 | No neighbor features | 7.9452 | 4.4647 |
| E4 | E4_2 | Unweighted neighbor features | 7.9456 | 4.4679 |
| E4 | E4_3 | Distance-weighted neighbors | 7.9456 | **4.4523 [TEST BEST]** |
| E4 | E4_4 | Wind + distance weighted neighbors | 7.9523 | 4.4675 |

### Macro R2 by station (score)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | -0.0554 | 0.1350 |
| E1 | E1_2 | Global XGBoost | 0.0717 | **0.4147 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 0.0365 | 0.3195 |
| E1 | E1_4 | Regional XGBoost | 0.2063 | 0.3041 |
| E1 | E1_5 | Global + local MLP residual | -24.5180 | -3.0389 |
| E2 | E2_1 | XGBoost | 0.0214 | **0.4185 [TEST BEST]** |
| E2 | E2_2 | LightGBM | 0.0151 | 0.4115 |
| E2 | E2_3 | GradientBoostingRegressor | -0.1404 | 0.3555 |
| E3 | E3_1 | Direct forecast | -0.2390 | **-0.0956 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | -0.2212 | -0.1022 |
| E4 | E4_1 | No neighbor features | -0.2144 | -0.0862 |
| E4 | E4_2 | Unweighted neighbor features | -0.2151 | -0.0818 |
| E4 | E4_3 | Distance-weighted neighbors | -0.2165 | -0.0791 |
| E4 | E4_4 | Wind + distance weighted neighbors | -0.2182 | **-0.0768 [TEST BEST]** |

### Primary macro RMSE (station x horizon) (ug/m3)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 9.7963 | 5.4830 |
| E1 | E1_2 | Global XGBoost | 9.8845 | **4.6373 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 10.3026 | 5.0538 |
| E1 | E1_4 | Regional XGBoost | 9.3894 | 4.9038 |
| E1 | E1_5 | Global + local MLP residual | 13.3344 | 7.0248 |
| E2 | E2_1 | XGBoost | 10.0661 | 4.6381 |
| E2 | E2_2 | LightGBM | 10.0840 | **4.5967 [TEST BEST]** |
| E2 | E2_3 | GradientBoostingRegressor | 10.6336 | 4.8943 |
| E3 | E3_1 | Direct forecast | 10.4573 | **5.8148 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 10.4537 | 5.8586 |
| E4 | E4_1 | No neighbor features | 10.4762 | 5.7604 |
| E4 | E4_2 | Unweighted neighbor features | 10.4786 | 5.7755 |
| E4 | E4_3 | Distance-weighted neighbors | 10.4814 | **5.7593 [TEST BEST]** |
| E4 | E4_4 | Wind + distance weighted neighbors | 10.4963 | 5.7682 |

### Worst-station RMSE (ug/m3)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 23.9174 | 13.7393 |
| E1 | E1_2 | Global XGBoost | 23.6018 | **10.4771 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 26.5046 | 12.6886 |
| E1 | E1_4 | Regional XGBoost | 21.9342 | 10.5942 |
| E1 | E1_5 | Global + local MLP residual | 257.9117 | 113.1200 |
| E2 | E2_1 | XGBoost | 23.2323 | 9.9755 |
| E2 | E2_2 | LightGBM | 23.0906 | **9.7599 [TEST BEST]** |
| E2 | E2_3 | GradientBoostingRegressor | 23.4795 | 10.8163 |
| E3 | E3_1 | Direct forecast | 24.1425 | **14.5507 [TEST BEST]** |
| E3 | E3_2 | Multi-output forecast | 24.0210 | 15.1189 |
| E4 | E4_1 | No neighbor features | 24.9775 | 14.7674 |
| E4 | E4_2 | Unweighted neighbor features | 25.3442 | 14.4902 |
| E4 | E4_3 | Distance-weighted neighbors | 25.2339 | 14.5171 |
| E4 | E4_4 | Wind + distance weighted neighbors | 25.6741 | **14.4786 [TEST BEST]** |

### RMSE degradation (horizon 1 to 7) (ug/m3; lower means a smaller increase)

| E group | Experiment | Method | Validation | Test |
|---|---|---|---:|---:|
| E1 | E1_1 | Local XGBoost | 5.9759 | 2.4303 |
| E1 | E1_2 | Global XGBoost | 7.6060 | **1.6820 [TEST BEST]** |
| E1 | E1_3 | Global + local tree residual | 8.4129 | 2.0085 |
| E1 | E1_4 | Regional XGBoost | 6.7064 | 2.1238 |
| E1 | E1_5 | Global + local MLP residual | 19.0250 | 9.9710 |
| E2 | E2_1 | XGBoost | 7.7606 | 2.1699 |
| E2 | E2_2 | LightGBM | 7.8682 | **2.0240 [TEST BEST]** |
| E2 | E2_3 | GradientBoostingRegressor | 8.3584 | 2.8239 |
| E3 | E3_1 | Direct forecast | 6.4216 | 2.7901 |
| E3 | E3_2 | Multi-output forecast | 6.4029 | **2.5924 [TEST BEST]** |
| E4 | E4_1 | No neighbor features | 6.3892 | 2.6657 |
| E4 | E4_2 | Unweighted neighbor features | 6.3964 | 2.7169 |
| E4 | E4_3 | Distance-weighted neighbors | 6.4242 | **2.6354 [TEST BEST]** |
| E4 | E4_4 | Wind + distance weighted neighbors | 6.4614 | 2.6952 |

## Validation Leader by the Primary Metric

| E group | Leader by Validation `primary_macro_rmse` | Validation score |
|---|---|---:|
| E1 | **E1_4 - Regional XGBoost** | 9.3894 |
| E2 | **E2_1 - XGBoost** | 10.0661 |
| E3 | **E3_2 - Multi-output forecast** | 10.4537 |
| E4 | **E4_1 - No neighbor features** | 10.4762 |

Test scores are reported separately and are not used to select the winner. E1.4 is the E1 validation leader; E1.2 has the lowest E1 test primary macro RMSE (4.6373), for descriptive reporting only.

Sources: each experiment's `metrics_overall.json`, `metrics_overall_test.json`, and `dataset_manifest.json`; metric definitions are in `experiment-stage-1/common/metrics.py`. E1.1-E1.5 use the fixed-common artifacts generated 2026-10-03.
