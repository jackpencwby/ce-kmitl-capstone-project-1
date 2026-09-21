# Additional PM2.5 literature and research directions

Checked 2026-09-19. These notes distinguish findings reported by authors from proposed experiments. The papers below address different tasks: retrospective estimation, atmospheric source attribution, and future forecasting. Their headline R²/RMSE values are not comparable without matching forecast origin, horizon, stations, dates, and available inputs.

**User clarification:** The user's XGBoost R² of approximately 0.9 is being compared with approximately 0.9 in other papers using different datasets. There has not been a controlled short-history versus long-history comparison. Therefore no data-sufficiency or training-length equivalence result is established. R² depends on the target variance and test population; equal values cannot rank models across datasets. The first step is to compare baselines on the same local data, forecast horizons, and splits.

## Six primary research sources

### 1. Damkliang and Chumnaul (2025): regional generalization remains unresolved

[Deep learning and statistical approaches for area-based PM2.5 forecasting in Hat Yai, Thailand](https://link.springer.com/article/10.1186/s40537-025-01079-9), Journal of Big Data 12, 36.

**Reported:** Daily univariate PM2.5 from Hat Yai, 2013–2023; ConvLSTM1D-BiLSTM reported MAE 2.05 and R² 0.68. The authors explicitly identify daily aggregation, a single main station, and omission of meteorology as limitations. They state that universal applicability across Thailand is not established. External validation used four nearby southern stations and a 14-day evaluation period.

**Inference for this project:** National station data enable a stronger question than another local architecture comparison: whether a pooled model transfers to unseen provinces and regions. Evaluate station-held-out future periods and regional errors. Do not equate good nearby-station performance with national generalization. Hourly forecasting would require genuinely hourly data; daily data cannot recover within-day peaks.

### 2. Inlaung et al. (2024): transboundary smoke is spatially heterogeneous

[Assessment of Transboundary PM2.5 from Biomass Burning in Northern Thailand Using the WRF-Chem Model](https://pmc.ncbi.nlm.nih.gov/articles/PMC11280843/), Toxics 12, 462. [Publisher PDF](https://mdpi-res.com/d_attachment/toxics/toxics-12-00462/article_deploy/toxics-12-00462.pdf?version=1719398717).

**Reported:** Compared simulations including all emissions with simulations excluding transboundary biomass-burning emissions. Chiang Rai and Mae Hong Son showed high transboundary contributions. Explicit limitations were ten monitoring stations, a 152-day study period, and uncertainty in model parameters and input data.

**Inference for this project:** Domestic hotspot counts alone can miss incoming smoke. Add lagged foreign hotspots, direction-sensitive fire exposure, and upstream station PM2.5. Compare domestic-only against cross-border features. This would test predictive value, not prove causal source contributions: source attribution in the paper depends on atmospheric simulation scenarios.

### 3. Nation-by-nation CMAQ source study (2024): hotspot quantity alone is insufficient

[A Nation-by-Nation Assessment of the Contribution of Southeast Asian Open Biomass Burning to PM2.5 in Thailand Using the Community Multiscale Air Quality-Integrated Source Apportionment Method Model](https://www.mdpi.com/2073-4433/15/11/1358), Atmosphere 15, 1358.

**Reported in publisher abstract:** The study simulated 2019. Myanmar burning dominated contributions in western/central upper northern areas, whereas domestic burning dominated the east during peak months. Despite high Lao emissions, meteorological transport carried PM2.5 eastward rather than into Thailand. Open burning had smaller overall influence in Bangkok than in upper northern Thailand.

**Scope limit:** This is a one-year source-attribution simulation, not a prospective ML forecast. Full text could not be retrieved in this search, so no additional author-stated limitation is asserted.

**Inference for this project:** Construct fire exposure using wind direction, distance, fire radiative power when available, and time lags. The research question is whether transport-informed features improve unseen-season and severe-episode forecasts beyond isotropic hotspot buffers. National-average improvements may hide strong regional differences.

### 4. Kawichai et al. (2025): historical reconstruction solves a different data gap

[Long-Term Retrospective Predicted Concentration of PM2.5 in Upper Northern Thailand Using Machine Learning Models](https://pmc.ncbi.nlm.nih.gov/articles/PMC11946178/), Toxics 13, 170.

**Reported:** Reconstructs historical PM2.5 for 2011–2020 using pollutants, meteorology, and hotspots. RF achieved RMSE 6.82 μg/m³ and R² 0.93. The motivating gap is missing historical PM2.5 coverage: monitoring began at different times across provinces. Model validation describes 10-fold cross-validation and a 70/30 partition.

**Inference for this project:** This supports research on sparse monitoring and historical reconstruction, but its performance is not a direct forecasting benchmark. Using observed contemporaneous pollutants or weather to reconstruct PM2.5 differs from predicting tomorrow using only information available today. Its ten-year span is therefore not evidence that a future-forecast model needs ten years or that two years is inferior.

### 5. CMAQ–LightGBM comparison (2026): known-region accuracy and spatial transfer differ

[A comparative study on CMAQ-based predictors and reanalysis AOD for improving PM2.5 estimation in Thailand using machine learning](https://doi.org/10.1016/j.atmosenv.2026.122006), Atmospheric Environment 375, 122006.

**Reported in publisher highlights/abstract:** Sparse monitoring motivates PM2.5 estimation. The study evaluates three strategies for spatial generalizability, incorporates source-decoupled CMAQ predictors, and reports that those predictors are most effective within known regions. Reanalysis AOD is presented as a practical alternative when CMAQ is unavailable.

**Scope limit:** This is estimation/mapping; full text could not be retrieved here. Do not infer a forecast horizon or numerical spatial-validation result from the abstract.

**Inference for this project:** National coverage is useful only if evaluated against the intended deployment setting. Separate known-station future forecasts from new-station forecasts and regional extrapolation. Satellite observations and reanalysis can support retrospective mapping but their availability at forecast issue time must be checked before using them in an operational forecasting experiment.

### 6. Panja et al. (2024/2025 preprint): optimize for severe pollution and uncertainty

[E-STGCN: Extreme Spatiotemporal Graph Convolutional Networks for Air Quality Forecasting](https://arxiv.org/abs/2411.12258). Preprint first submitted November 2024, revised June 2025; [author publication record](https://anubhab17.github.io/anubhabbiswas.github.io/publication/2024-11-18-e-stgcn-air-quality) identifies a 2026 JRSS A publication, [DOI](https://doi.org/10.1093/jrsssa/qnag010). Claims here use the accessible preprint.

**Reported:** Models extreme pollution with EVT-guided temporal learning and graph-based spatial structure, using PM2.5, PM10, and NO2 at 37 Delhi monitoring stations. Conformal prediction provides probabilistic intervals. Motivation includes nonstationarity, spatial dependence, and extremes that ordinary models may fail to represent.

**Inference for this project:** This is a method precedent, not proof of benefits in Thailand. Begin with tail-focused XGBoost objectives or a threshold-event classifier plus regression, before adopting a graph network. Evaluate error on high-pollution days, missed episodes, false alarms, and interval coverage/width by season and region. Adding conformal prediction alone is not novel; testing calibration under Thai seasonal and spatial shifts can be a defensible contribution. Ordinary exchangeability-based coverage claims do not automatically transfer to dependent, shifting station time series.

## Guard against an overstated geographic research gap

[Spatiotemporal Forecasting of PM2.5 Using Machine Learning and GIS in Northeastern Thailand](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5388146) is a 2025 SSRN preprint describing meteorological/temporal predictors and RF, XGBoost, and SVR for 2017–2023. It reports XGBoost R² 0.82 and RMSE 8.1 μg/m³. It is not sufficient evidence of a rigorous forecasting benchmark, but it prevents an unqualified claim that nobody has studied multivariate PM2.5 prediction in the Northeast. Verify publication status, methods, and forecast-time feature availability before treating it as a main comparator. [CMU institutional research record](https://datascience.cmu.ac.th/library/articles/90) is another item to classify carefully as estimation versus forecasting.

## Proposed research focus for the available dataset

These are proposed experiments, not findings established by the papers.

**Recommended question:** Can transport-informed features improve severe-episode and spatial-transfer performance of a national PM2.5 forecast when training history is limited?

Use the present XGBoost model as the baseline. Add features in controlled stages: local weather and target lags; isotropic fire buffers; lagged, wind-weighted domestic and cross-border fire exposure; upstream PM2.5. Use forecasts issued at a defined daily cutoff for horizons of 1, 3, and 7 days. Every predictor must have been observable by that cutoff. If future observed weather is used, report that separately as an oracle-input experiment.

**Core evaluation:** Freeze identical test dates/stations for all models. Compare rolling training histories such as 6, 12, 18, and 24 months with expanding history where available. Repeat over feasible forecast origins, preserving complete seasonal blocks. Keep a stable station cohort for the principal data-length comparison; separately evaluate newly added stations. Otherwise a changing monitoring network confounds training length with spatial coverage. Report overall MAE/RMSE, station-macro MAE, region/season results, tail MAE/bias, exceedance-event recall and precision, and skill over persistence. Estimate paired uncertainty with temporal blocks rather than treating every station-day as independent.

**Training length is an optional future experiment:** The current cross-paper R² comparison says nothing about how much history is sufficient. In a future controlled experiment, similar aggregate scores could reflect sufficient seasonal information, strong persistence, redundant older observations, or outdated regimes. Specify a practical error-difference margin in advance and quantify uncertainty in the paired difference. The dataset's limited history restricts how many independent seasonal tests can be made.

**Feasible contribution even if aggregate RMSE barely changes:** Demonstrate fewer missed severe episodes, better performance in held-out provinces, reliable uncertainty during burning periods, or equivalent skill with a shorter and cheaper training history. Select one primary contribution and make the others supporting analyses. A new architecture is optional.

## Additional methodological pointers verified by the coordinating researcher

- [Lat Krabang forecasting study, published 31 July 2026](https://ijtech.eng.ui.ac.id/article/view/8040), [full paper](https://ijtech.eng.ui.ac.id/download/article/8040): in a one-hour forecast setting, the coordinating researcher reports RF RMSE 5.825 versus best BiRNN 5.712, strong lag-one importance, and weaker Transformer performance. This is a useful baseline-selection precedent; it does not establish model rankings for national daily forecasting.
- [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170), NeurIPS 2021: methodological starting point for adaptive prediction intervals. Do not interpret aggregate calibration as a guarantee of conditional coverage within every region, season, or extreme episode.

The recommended scope is transport-informed XGBoost with severe-event evaluation at 3- and 7-day horizons; training-length learning curves can be secondary. Another viable scope is spatial pooling for stations with short histories. Neither is claimed to be the first application without a further targeted novelty review.
