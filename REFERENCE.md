# REFERENCE.md — Temporal Fusion Transformer (TFT) with Darts

เอกสารอ้างอิงสำหรับงาน multi-horizon time series forecasting ด้วย TFT
รวมสาระจาก 2 แหล่ง:

1. **Paper**: Lim, Arık, Loeff, Pfister — *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting* — https://arxiv.org/abs/1912.09363 (v3)
2. **Implementation**: `darts.models.forecasting.tft_model.TFTModel` (Darts v0.46.1) — https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html

> **สำหรับ Claude Code**: อ่านส่วน `§3 Data Contract`, `§5 API Signature`, `§8 Pitfalls` ก่อนเขียนโค้ดทุกครั้ง
> พารามิเตอร์ที่ผิดบ่อยที่สุดคือ `future_covariates` (บังคับ) และ `output_chunk_length` vs `n`

---

## 1. TFT คืออะไร และควรใช้เมื่อไหร่

TFT เป็นสถาปัตยกรรม attention-based สำหรับพยากรณ์หลายช่วงเวลาข้างหน้าพร้อมกัน (direct multi-horizon)
ที่ออกแบบมาเพื่อรับ input 3 ชนิดที่ต่างกันโดยตรง และให้ผลลัพธ์ที่ตีความได้

**ใช้ TFT เมื่อ:**
- มีทั้ง static metadata, ตัวแปรที่รู้ค่าล่วงหน้า (วันหยุด/โปรโมชัน) และตัวแปรที่รู้เฉพาะอดีต
- ต้องการ prediction interval (quantile) ไม่ใช่แค่ค่าจุดเดียว
- ต้องการอธิบายได้ว่าตัวแปรไหนสำคัญ / โมเดลมองย้อนหลังไปที่ช่วงไหน
- มีหลาย entity (หลายร้าน/หลายสินค้า/หลาย sensor) → train โมเดลเดียวแบบ global

**อย่าใช้ TFT เมื่อ:**
- ข้อมูลสั้นมาก (< ~1,000 จุด ต่อ series) → เริ่มที่ `LinearRegressionModel`, `LightGBMModel`, `NHiTS` ก่อน
- ต้องการ baseline เร็ว ๆ → TFT เทรนช้าและมี hyperparameter เยอะ
- ไม่มี covariate เลย และ pattern เป็น seasonality ตรงไปตรงมา

**Benchmark จากเปเปอร์**: TFT ชนะ DeepAR, DSSM, ConvTrans, Seq2Seq, MQRNN ทุก dataset
โดยดีกว่าอันดับสองประมาณ 7% (P50) และ 9% (P90)

---

## 2. สถาปัตยกรรม (สรุปจากเปเปอร์ §4)

ลำดับการไหลของข้อมูล:

```
Static covariates ──► Static Variable Selection ──► GRN encoders ──► c_s, c_e, c_c, c_h
                                                                      │
Past inputs  ──► Variable Selection (ใช้ c_s) ──┐                     │
                                                ├─► LSTM Encoder/Decoder (init ด้วย c_c, c_h)
Known future ──► Variable Selection (ใช้ c_s) ──┘                     │
                                                                      ▼
                                          Gated skip connection + LayerNorm
                                                      │
                                          Static Enrichment GRN (ใช้ c_e)
                                                      │
                                     Interpretable Multi-Head Attention (masked)
                                                      │
                                          Gate + LayerNorm + Position-wise GRN
                                                      │
                                          Gated skip ข้ามทั้ง transformer block
                                                      │
                                          Linear → Quantile outputs
```

### องค์ประกอบหลัก 5 อย่าง

| องค์ประกอบ | หน้าที่ | ทำไมสำคัญ (จาก ablation §6.6) |
|---|---|---|
| **GRN** (Gated Residual Network) | บล็อกพื้นฐาน: ELU + Linear + GLU + LayerNorm + skip | ตัดออก → P90 loss +1.9% เฉลี่ย, +4.1% บนข้อมูลเล็ก/noisy |
| **Variable Selection Network** | ให้น้ำหนัก softmax แต่ละตัวแปร รายจุดเวลา (instance-wise) | ตัดออก → P90 +4.1% เฉลี่ย |
| **Static Covariate Encoder** | สร้าง context 4 ตัว (`c_s, c_e, c_c, c_h`) ฉีดเข้าหลายจุด | ตัดออก → P90 +2.6% เฉลี่ย |
| **Seq2Seq layer (LSTM)** | จับ local pattern แทน positional encoding | ตัดออก → P90 +6% เฉลี่ย, บางชุด +20% |
| **Interpretable Multi-Head Attention** | จับ long-range dependency + ตีความได้ | ตัดออก → P90 +6% เฉลี่ย |

### สูตรสำคัญ

**GRN** (สมการ 2–4):
```
GRN(a, c) = LayerNorm(a + GLU(η₁))
η₁ = W₁η₂ + b₁
η₂ = ELU(W₂a + W₃c + b₂)     # ถ้าไม่มี context ให้ c = 0
```

**GLU** (สมการ 5) — เป็นตัวที่ทำให้โมเดล "ข้าม" layer ได้ถ้าไม่จำเป็น:
```
GLU(γ) = σ(W₄γ + b₄) ⊙ (W₅γ + b₅)
```

**Interpretable Multi-Head Attention** (สมการ 13–16) — ต่างจาก Transformer ปกติตรงที่
**แชร์ค่า V ทุก head แล้วเฉลี่ย attention weight** ทำให้ weight ตีความได้ตรง ๆ:
```
InterpretableMultiHead(Q,K,V) = { (1/H) Σₕ A(QW_Q⁽ʰ⁾, KW_K⁽ʰ⁾) } · V·W_V · W_H
```

**Quantile Loss** (สมการ 24–25):
```
QL(y, ŷ, q) = q·max(0, y−ŷ) + (1−q)·max(0, ŷ−y)
L = Σ Σ Σ QL(...) / (M · τ_max)
```

**q-Risk สำหรับ evaluate** (สมการ 26) — คือ quantile loss ที่ normalize ด้วยผลรวม |y|:
```
q-Risk = 2·Σ QL(yₜ, ŷ(q,t−τ,τ), q) / Σ |yₜ|
```

---

## 3. Data Contract — input 3 ชนิด (สำคัญที่สุด)

เปเปอร์แบ่ง input เป็น 3 ประเภท ซึ่ง map ตรงกับ Darts:

| ประเภทในเปเปอร์ | สัญลักษณ์ | ความหมาย | พารามิเตอร์ใน Darts | ช่วงเวลาที่ต้องมีข้อมูล |
|---|---|---|---|---|
| Static covariates | `s_i` | ไม่เปลี่ยนตามเวลา เช่น รหัสร้าน, ภูมิภาค, หมวดสินค้า | `series.static_covariates` (แนบกับ `TimeSeries`) | — |
| Observed (past) inputs | `z_{i,t}` | รู้เฉพาะอดีต เช่น ยอดขายคู่แข่ง, จำนวนลูกค้า, ราคาน้ำมัน | `past_covariates` | ต้องครอบคลุม `input_chunk_length` ก่อนจุดพยากรณ์ |
| Known future inputs | `x_{i,t}` | รู้ล่วงหน้า เช่น วันในสัปดาห์, วันหยุด, โปรโมชันที่วางแผนไว้ | `future_covariates` | ต้องครอบคลุมถึง `output_chunk_length` หลังจุดพยากรณ์ |

**กฎเหล็ก 3 ข้อ:**

1. **`future_covariates` เป็นสิ่งจำเป็น (mandatory)** — TFT ใช้มันเป็น query ของ attention
   ถ้าไม่มีข้อมูลจริง ต้องใช้ทางออกใดทางหนึ่ง:
   - `add_relative_index=True` (ใส่ positional value ปกติที่ normalize ด้วย `input_chunk_length`)
   - `add_encoders={...}` ให้ Darts สร้าง calendar features อัตโนมัติ
2. **Static covariates ต้องเป็นตัวเลข** — `TorchForecastingModel` รองรับเฉพาะตัวเลข
   ใช้ `StaticCovariatesTransformer` แปลง categorical ก่อน แล้วประกาศใน `categorical_embedding_sizes`
3. **ต้อง scale ข้อมูล** — เปเปอร์ใช้ z-score normalization แยกตาม entity และ log-transform สำหรับ target ที่เป็น sales/volatility
   ใน Darts ใช้ `Scaler` (fit บน train เท่านั้น)

---

## 4. Mapping: เปเปอร์ ↔ Darts

| แนวคิดในเปเปอร์ | ชื่อในเปเปอร์ | พารามิเตอร์ Darts | ค่า default |
|---|---|---|---|
| Look-back window | `k` | `input_chunk_length` | ต้องระบุ |
| Forecast horizon | `τ_max` | `output_chunk_length` | ต้องระบุ |
| State size / hidden size | `d_model` | `hidden_size` | `16` |
| Number of attention heads | `m_H` | `num_attention_heads` | `4` |
| Dropout rate | — | `dropout` | `0.1` |
| จำนวน LSTM layers | — | `lstm_layers` | `1` |
| Decoder masking (causal) | — | `full_attention=False` | `False` |
| Quantile set `Q = {0.1, 0.5, 0.9}` | — | `likelihood=QuantileRegression(quantiles=[...])` | QuantileRegression |
| Minibatch size | — | `batch_size` (kwargs) | `32` |
| Learning rate | — | `optimizer_kwargs={"lr": ...}` | Adam default |
| Max gradient norm | — | `pl_trainer_kwargs={"gradient_clip_val": ...}` | ไม่ตั้ง |
| GRN feed-forward | — | `feed_forward="GatedResidualNetwork"` | เหมือนเปเปอร์ |
| ขนาด hidden ของตัวแปร continuous | — | `hidden_continuous_size` | `8` |
| entity embeddings | — | `categorical_embedding_sizes` | `None` |

**พารามิเตอร์ที่ Darts เพิ่มเข้ามา (ไม่มีในเปเปอร์):**

| พารามิเตอร์ | ความหมาย |
|---|---|
| `output_chunk_shift` | เว้นช่องว่างระหว่าง input chunk กับ output chunk (ถ้า > 0 จะทำ autoregression ไม่ได้) |
| `feed_forward` | เลือก GLU variant: `"GLU"`, `"Bilinear"`, `"ReGLU"`, `"GEGLU"`, `"SwiGLU"`, `"ReLU"`, `"GELU"` หรือ `"GatedResidualNetwork"` (ต้นฉบับ) |
| `norm_type` | `"LayerNorm"` (default), `"RMSNorm"`, `"LayerNormNoBias"` หรือ custom `nn.Module` |
| `skip_interpolation` | `True` = แทน interpolation ด้วย linear projection ใน VSN → เทรน/inference เร็วขึ้น แต่เสีย permutation ใน embedding space |
| `use_reversible_instance_norm` | RINorm กัน distribution shift (ใช้กับ target เท่านั้น ไม่ใช้กับ covariates) |
| `add_relative_index` | เติม positional value ให้ future covariates เพื่อให้ใช้โมเดลได้โดยไม่ต้องส่ง future_covariates |
| `enable_finetuning` | freeze/unfreeze พารามิเตอร์ตาม pattern เช่น `{"unfreeze": ["...*"]}` |

---

## 5. API Signature

```python
TFTModel(
    input_chunk_length,               # int  — k (encoder length)
    output_chunk_length,              # int  — τ_max (decoder length)
    output_chunk_shift=0,
    hidden_size=16,                   # int | list[int] — hyperparameter หลัก
    lstm_layers=1,
    num_attention_heads=4,
    full_attention=False,
    feed_forward="GatedResidualNetwork",
    dropout=0.1,
    hidden_continuous_size=8,
    categorical_embedding_sizes=None, # {"col": n} หรือ {"col": (n, emb_dim)}
    skip_interpolation=False,
    loss_fn=None,                     # ใช้เมื่อตั้ง likelihood=None (deterministic)
    likelihood=None,                  # default = QuantileRegression
    norm_type="LayerNorm",
    use_static_covariates=True,
    **kwargs                          # ส่งต่อไป PL Module / Trainer / TorchForecastingModel
)
```

**kwargs ที่ใช้บ่อย:**
`batch_size`, `n_epochs` (default 100), `model_name`, `work_dir`, `save_checkpoints`,
`force_reset`, `log_tensorboard`, `random_state`, `optimizer_cls`, `optimizer_kwargs`,
`lr_scheduler_cls`, `lr_scheduler_kwargs`, `add_encoders`, `pl_trainer_kwargs`,
`torch_metrics`, `use_reversible_instance_norm`, `show_warnings`

**เมธอดหลัก:**
`fit()`, `predict()`, `historical_forecasts()`, `backtest()`, `residuals()`, `gridsearch()`,
`lr_find()`, `save()`, `load()`, `load_from_checkpoint()`, `load_weights()`,
`scale_batch_size()`, `to_onnx()`, `untrained_model()`, `reset_model()`,
`generate_fit_encodings()`, `generate_predict_encodings()`

---

## 6. Template โค้ดมาตรฐาน

```python
import numpy as np
import torch
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
from darts.utils.likelihood_models.torch import QuantileRegression
from darts.metrics import mql, mape, rmse
from pytorch_lightning.callbacks import EarlyStopping

QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
INPUT_CHUNK = 168          # k
OUTPUT_CHUNK = 24          # τ_max

# ── 1. เตรียมข้อมูล ─────────────────────────────────────────────
# series, past_cov, future_cov เป็น TimeSeries (หรือ list ของ TimeSeries สำหรับหลาย entity)
train, val = series.split_after(0.8)

scaler_y = Scaler()
train_s = scaler_y.fit_transform(train)      # fit บน train เท่านั้น
val_s   = scaler_y.transform(val)

scaler_past = Scaler();   past_s   = scaler_past.fit_transform(past_cov)
scaler_fut  = Scaler();   future_s = scaler_fut.fit_transform(future_cov)

# static covariates ที่เป็น categorical
static_tf = StaticCovariatesTransformer()
train_s = static_tf.fit_transform(train_s)
val_s   = static_tf.transform(val_s)

# ── 2. สร้างโมเดล ───────────────────────────────────────────────
early_stop = EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-4, mode="min")

model = TFTModel(
    input_chunk_length=INPUT_CHUNK,
    output_chunk_length=OUTPUT_CHUNK,
    hidden_size=64,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    hidden_continuous_size=16,
    batch_size=64,
    n_epochs=100,
    likelihood=QuantileRegression(quantiles=QUANTILES),
    optimizer_kwargs={"lr": 1e-3},
    lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
    lr_scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 5},
    add_encoders={
        "cyclic": {"future": ["month", "dayofweek", "hour"]},
        "datetime_attribute": {"future": ["dayofweek", "hour"]},
        "position": {"past": ["relative"], "future": ["relative"]},
        "transformer": Scaler(),
    },
    pl_trainer_kwargs={
        "accelerator": "gpu", "devices": [0],
        "gradient_clip_val": 1.0,           # = Max Gradient Norm ในเปเปอร์
        "callbacks": [early_stop],
    },
    model_name="tft_run",
    save_checkpoints=True,
    force_reset=True,
    random_state=42,
)

# ── 3. เทรน ─────────────────────────────────────────────────────
model.fit(
    series=train_s,
    past_covariates=past_s,
    future_covariates=future_s,
    val_series=val_s,
    val_past_covariates=past_s,
    val_future_covariates=future_s,
    verbose=True,
)

# ── 4. พยากรณ์แบบ probabilistic ─────────────────────────────────
pred = model.predict(
    n=OUTPUT_CHUNK,          # n <= output_chunk_length → ไม่ทำ autoregression
    num_samples=200,         # ต้อง >> 1 เพราะโมเดลเป็น probabilistic
    past_covariates=past_s,
    future_covariates=future_s,
)
pred = scaler_y.inverse_transform(pred)

# quantile ที่ต้องการ
p10 = pred.quantile_timeseries(0.1)
p50 = pred.quantile_timeseries(0.5)
p90 = pred.quantile_timeseries(0.9)

# ── 5. ประเมินผล ────────────────────────────────────────────────
score_p50 = mql(val, pred, q=0.5)
score_p90 = mql(val, pred, q=0.9)

# backtest แบบไม่ retrain (เร็ว)
bt = model.backtest(
    series=series_s,
    past_covariates=past_s,
    future_covariates=future_s,
    start=0.8,
    forecast_horizon=OUTPUT_CHUNK,
    stride=OUTPUT_CHUNK,
    retrain=False,
    num_samples=200,
    metric=[mql, rmse],
    metric_kwargs=[{"q": 0.5}, {}],
)
```

---

## 7. Interpretability (จุดขายหลักของ TFT)

เปเปอร์นำเสนอ use case 3 แบบ (§7) — Darts มี `TFTExplainer` รองรับ

```python
from darts.explainability.tft_explainer import TFTExplainer

explainer = TFTExplainer(
    model,
    background_series=train_s,
    background_past_covariates=past_s,
    background_future_covariates=future_s,
)
result = explainer.explain()

explainer.plot_variable_selection(result)   # ความสำคัญของตัวแปร (static / past / future)
explainer.plot_attention(result, plot_type="all")   # attention pattern
```

### 7.1 Variable importance
มาจากน้ำหนัก softmax ของ Variable Selection Network (`v_χt` ในสมการ 8)
รวบรวมทั่ว test set แล้วดู percentile ที่ 10/50/90

ตัวอย่างผลจากเปเปอร์ (Retail dataset, median weight):
- Static: Item Num 0.230, Store Num 0.161, Class 0.156
- Past: Log Sales 0.324, National Holiday 0.138, Month 0.122
- Future: National Holiday 0.220, On-promotion 0.170, Month 0.155

สำหรับ Electricity: `Hour of Day` มีน้ำหนักสูงกว่าตัว target เอง (0.462 vs 0.359)

### 7.2 Persistent temporal patterns
ดู attention weight `α(t, n, τ)` → เห็น seasonality / lag effect โดยไม่ต้อง hard-code
เปเปอร์พบ spike รายวันชัดเจนใน Electricity และ Traffic, แบบรายสัปดาห์ที่อ่อนกว่าใน Retail

**ใช้ปรับโมเดล**: ถ้า attention peak อยู่ที่ขอบซ้ายสุดของ lookback window → เพิ่ม `input_chunk_length`

### 7.3 Regime / event detection
เทียบ attention vector แต่ละจุดเวลากับ pattern เฉลี่ย ด้วยระยะทางจาก Bhattacharyya coefficient:
```
κ(p, q) = sqrt(1 − Σⱼ sqrt(pⱼqⱼ))
dist(t) = Σ_τ κ(ᾱ(τ), α(t,τ)) / τ_max
```
เปเปอร์ใช้เกณฑ์ `dist(t) > 0.3` เพื่อชี้ regime สำคัญ (เช่น วิกฤตการเงินปี 2008 บน S&P 500)

---

## 8. Pitfalls — ข้อผิดพลาดที่พบบ่อย

| ปัญหา | สาเหตุ | วิธีแก้ |
|---|---|---|
| `future_covariates` is required | TFT ต้องใช้ future input เป็น attention query | ตั้ง `add_relative_index=True` หรือใส่ `add_encoders` |
| พยากรณ์ออกมาเป็นค่าเดียว ไม่มีช่วงความเชื่อมั่น | ลืมตั้ง `num_samples` | `predict(n, num_samples=200)` |
| `n > output_chunk_length` แล้วผลแย่ | โมเดลทำ autoregression ต่อหลายรอบ → error สะสม | ตั้ง `output_chunk_length` ให้เท่า horizon ที่ต้องการจริง |
| covariates ไม่ครอบคลุมช่วงพยากรณ์ | `future_covariates` ต้องยาวถึง `t + output_chunk_length` | ขยาย future covariates ล่วงหน้า |
| static covariate เป็น string → error | Torch models รองรับเฉพาะตัวเลข | `StaticCovariatesTransformer` |
| loss ไม่ลด / NaN | ไม่ได้ scale ข้อมูล หรือ lr สูงเกิน | `Scaler` + `lr_find()` + `gradient_clip_val` |
| overfitting เร็ว | dataset เล็ก, hidden_size ใหญ่ | เพิ่ม `dropout` (0.3–0.5), ลด `hidden_size`, ใช้ EarlyStopping |
| เทรนช้ามาก | VSN interpolation + hidden ใหญ่ | `skip_interpolation=True`, ลด `input_chunk_length`, `scale_batch_size()` |
| ตั้ง `output_chunk_shift > 0` แล้ว autoregression พัง | เป็นข้อจำกัดตามสเปค | ตั้ง `n <= output_chunk_length` |
| ผลไม่ reproducible | ไม่ได้ล็อก seed | `random_state=42` |
| Data leakage ใน backtest | Scaler fit บนข้อมูลทั้งหมด | fit scaler บน train เท่านั้น หรือใช้ `data_transformers` ใน `historical_forecasts()` |

---

## 9. Hyperparameter search space

**ช่วงที่เปเปอร์ใช้ค้นหา (random search):**

| Hyperparameter | ช่วงค้นหา |
|---|---|
| State size (`hidden_size`) | 10, 20, 40, 80, 160, 240, 320 |
| Dropout rate | 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9 |
| Minibatch size | 64, 128, 256 |
| Learning rate | 0.0001, 0.001, 0.01 |
| Max gradient norm | 0.01, 1.0, 100.0 |
| Number of heads | 1, 4 |

**ค่าที่ดีที่สุดที่เปเปอร์พบ (Table 1):**

| | Electricity | Traffic | Retail | Volatility |
|---|---|---|---|---|
| `input_chunk_length` (k) | 168 | 168 | 90 | 252 |
| `output_chunk_length` (τ_max) | 24 | 24 | 30 | 5 |
| `dropout` | 0.1 | 0.3 | 0.1 | 0.3 |
| `hidden_size` | 160 | 320 | 240 | 160 |
| `num_attention_heads` | 4 | 4 | 4 | 1 |
| `batch_size` | 64 | 128 | 128 | 64 |
| learning rate | 0.001 | 0.001 | 0.001 | 0.01 |
| gradient clip | 0.01 | 100 | 100 | 0.01 |

**Heuristic เริ่มต้น:**
- `input_chunk_length` ≈ 3–7 เท่าของ `output_chunk_length` หรือ 1 seasonal cycle เต็ม
- เริ่มที่ `hidden_size=64`, `num_attention_heads=4`, `dropout=0.1` แล้วค่อยขยาย
- `hidden_continuous_size` ≈ `hidden_size / 4`
- ใช้ attention layer เดียว (เปเปอร์เลือกแบบนี้เพื่อรักษา explainability)

**ต้นทุนการคำนวณ**: เปเปอร์รายงานว่าโมเดลที่ดีที่สุดของ Electricity เทรนบน V100 ตัวเดียวใช้เวลา
ประมาณ 6 ชั่วโมง (~52 นาที/epoch) และ inference บน validation set 50,000 ตัวอย่างใช้ 8 นาที

---

## 10. Preprocessing ตามแนวเปเปอร์

- **Normalization**: z-score แยกตาม entity สำหรับ real-valued inputs ทุกตัว
- **Target transform**: log-transform สำหรับ sales และ realized volatility
- **Missing data**: resample เป็นความถี่สม่ำเสมอ แล้ว impute ด้วยค่าล่าสุด (`MissingValuesFiller`)
  พร้อมเพิ่ม flag `open` บอกว่าวันนั้นมีข้อมูลจริงหรือไม่
- **Calendar features**: day-of-week, day-of-month, week-of-year, month, hour-of-day เป็น **categorical**
- **Time index**: จำนวน step นับจากจุดเริ่มต้น — เป็น real-valued input
- **Holidays**: แยกเป็นตัวแปรคนละตัว (national / regional / local) ไม่รวมเป็นตัวเดียว
- **Entity definition**: ใน Retail เปเปอร์นับ (product × store) เป็น 1 entity → 135k entities

---

## 11. Checklist ก่อนส่งงาน

- [ ] `future_covariates` มีจริง หรือตั้ง `add_relative_index` / `add_encoders` แล้ว
- [ ] Scaler fit บน train split เท่านั้น (ไม่มี leakage)
- [ ] static covariates เป็นตัวเลขทั้งหมด
- [ ] `output_chunk_length` = horizon ที่ต้องการจริง และเรียก `predict(n <= output_chunk_length)`
- [ ] `num_samples > 1` เมื่อใช้ likelihood แบบ probabilistic
- [ ] มี `val_series` + EarlyStopping
- [ ] `random_state` ถูกตั้ง
- [ ] inverse_transform ผลลัพธ์ก่อนรายงาน metric
- [ ] เทียบกับ baseline (`NaiveSeasonal`, `LinearRegressionModel`) แล้ว
- [ ] รัน `TFTExplainer` เพื่อ sanity check ว่าตัวแปรที่สำคัญสมเหตุสมผล

---

## 12. ลิงก์

- Paper (arXiv): https://arxiv.org/abs/1912.09363
- Darts TFTModel API: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html
- Darts TFT example notebook: https://unit8co.github.io/darts/examples/13-TFT-examples.html
- Darts TFTExplainer: https://unit8co.github.io/darts/generated_api/darts.explainability.tft_explainer.html
- Darts metrics: https://unit8co.github.io/darts/generated_api/darts.metrics.html
- Official TFT (TensorFlow): https://github.com/google-research/google-research/tree/master/tft
- pytorch-forecasting TFT (ที่ Darts ยืมโครงสร้างภายในมา): https://pytorch-forecasting.readthedocs.io/en/latest/models.html
- GLU Variants (feed_forward options): https://arxiv.org/abs/2002.05202
- RINorm: https://openreview.net/forum?id=cGDAkQo1C0p