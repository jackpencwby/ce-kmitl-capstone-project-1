# แผนการทดลองโมเดลพยากรณ์ PM2.5 ล่วงหน้า 1–7 วัน

## 1. วัตถุประสงค์

เอกสารนี้กำหนดแผนการทดลองสำหรับเปรียบเทียบวิธีพยากรณ์ค่า PM2.5 รายวันล่วงหน้า 7 วัน โดยตอบคำถามหลัก 4 ด้าน

1. ควรฝึกโมเดลแยกสถานี รวมทุกสถานี รวมแล้วปรับเฉพาะสถานี หรือแยกตามภูมิภาค
2. XGBoost, LightGBM และ Gradient Boosting Regressor ให้ผลแตกต่างกันอย่างไร
3. การสร้างโมเดลแยกตามระยะพยากรณ์กับการพยากรณ์หลายวันพร้อมกัน วิธีใดเหมาะกว่า
4. ข้อมูล PM2.5 จากสถานีข้างเคียง รวมถึงระยะทางและทิศทางลม ช่วยเพิ่มความแม่นยำหรือไม่

ผลลัพธ์สุดท้ายต้องตอบได้ทั้งว่า **โมเดลใดแม่นที่สุด** และ **องค์ประกอบใดทำให้ผลพยากรณ์ดีขึ้น** โดยไม่ใช้ข้อมูลชุดทดสอบในการเลือกโมเดล

---

## 2. นิยามงานพยากรณ์

- หน่วยข้อมูล: รายวัน แยกตามสถานีตรวจวัด
- ตัวแปรเป้าหมาย: `pm25`
- ระยะพยากรณ์: `t+1` ถึง `t+7`
- ตัวแปรเป้าหมายของแต่ละระยะ:

```text
y_t1 = PM2.5 ของวัน t+1
y_t2 = PM2.5 ของวัน t+2
...
y_t7 = PM2.5 ของวัน t+7
```

- หน่วยรายงานผล: µg/m³
- แถวข้อมูล ณ วัน `t` ใช้ได้เฉพาะข้อมูลที่เกิดขึ้นไม่เกินวัน `t`
- ห้ามเติมค่าที่หายของ target และห้ามใช้ข้อมูลอนาคตในการสร้าง lag, rolling, spatial หรือการแทนค่าที่หาย

### 2.1 วิธีสร้าง Target

สร้าง target ภายในแต่ละสถานีและเรียงตามเวลาเสมอ

```python
for h in range(1, 8):
    df[f"target_t{h}"] = (
        df.groupby("station_id")["pm25"].shift(-h)
    )
```

แถวที่ target ของ horizon ที่กำลังทดลองเป็นค่าว่างต้องถูกตัดออก สำหรับ multi-output ให้ใช้เฉพาะแถวที่มี target ครบทั้ง 7 วัน เพื่อให้เปรียบเทียบกับ direct forecasting บน cohort เดียวกันได้

---

## 3. กติกากลางที่ต้องคงที่ทุกการทดลอง

เพื่อให้ผลเปรียบเทียบกันได้อย่างยุติธรรม ให้ตรึงรายการต่อไปนี้ตลอดโครงการ

| รายการ | กติกา |
|---|---|
| Dataset version | ใช้ไฟล์ master และ preprocessing version เดียวกัน |
| Time split | ใช้วันเริ่ม validation/test เดียวกันทุกโมเดล |
| Evaluation rows | ใช้สถานี วันที่ และ target cohort เดียวกันเมื่อเปรียบเทียบกัน |
| Feature groups | คงที่ เว้นแต่เป็นการทดลอง feature/spatial โดยตรง |
| Missing handling | fit/ตัดสินใจจาก train เท่านั้น และใช้กฎเดียวกันกับ validation/test |
| Random seed | ค่าเริ่มต้น `42`; finalists ประเมินซ้ำหลาย seed ถ้า algorithm มี randomness |
| Metrics | สูตรและ aggregation เดียวกัน |
| Hardware/software | บันทึก Python และ package versions ทุก run |
| Test set | lock ไว้จนกว่าจะเลือกโมเดลสุดท้ายเสร็จ |

หากใช้ค่าจาก configuration ปัจจุบัน ให้กำหนด

```yaml
validation_start: 2026-05-08
test_start: 2026-07-07
forecast_horizons: [1, 2, 3, 4, 5, 6, 7]
seed: 42
```

หาก dataset รุ่นล่าสุดกำหนดวันแบ่งชุดข้อมูลต่างจากนี้ ให้เปลี่ยนที่ configuration กลางเพียงแห่งเดียวก่อนเริ่มทดลอง และห้ามแก้วันแบ่งข้อมูลตามผลของแต่ละโมเดล

### 3.1 Feature baseline

Feature baseline ควรประกอบด้วยกลุ่มต่อไปนี้เท่าที่มีใน master dataset

- PM2.5 ของสถานีเป้าหมาย: lag 1, 3, 7 และ 30 วัน
- สภาพอากาศ: อุณหภูมิ ความชื้น ความกดอากาศ ความเร็วลม และทิศทางลม
- ตัวแปรสภาพอากาศแบบ lag/rolling ที่สร้างโดยไม่มองอนาคต
- CO, AOD และ hotspot ตาม availability ของข้อมูล
- ปฏิทิน: วัน เดือน ปี วันหยุด และ cyclical encoding
- metadata: `station_id`, latitude, longitude, elevation และ region เมื่อเหมาะสมกับ strategy

Rolling feature ณ วัน `t` ต้อง shift ก่อน rolling หากนิยามคือข้อมูลก่อนวันปัจจุบัน เช่น

```python
pm25_lagged = df.groupby("station_id")["pm25"].shift(1)
df["pm25_roll_mean_7"] = (
    pm25_lagged.groupby(df["station_id"]).rolling(7).mean()
    .reset_index(level=0, drop=True)
)
```

### 3.2 การจัดการ categorical features

- XGBoost/Gradient Boosting Regressor: ใช้ one-hot encoding ที่ fit จาก train เท่านั้น
- LightGBM: ใช้ categorical feature แบบ native หรือ one-hot ให้กำหนดแนวทางเดียวและบันทึกไว้
- category ที่ไม่เคยพบใน train ต้องมีค่า `unknown`
- Global และ Regional model ต้องมี `station_id`; Regional model ต้องมี `region_id`
- Local model ไม่จำเป็นต้องมี `station_id` เพราะหนึ่งโมเดลเห็นเพียงสถานีเดียว

### 3.3 Compute และ Device policy

- XGBoost: ใช้ CUDA ผ่าน `device="cuda"` + `tree_method="hist"` (เวอร์ชัน >= 2.0)
- LightGBM: ใช้ GPU build ได้ (`device_type="gpu"`/CUDA) ถ้า environment รองรับ มิฉะนั้น fallback CPU
- Gradient Boosting Regressor: CPU เท่านั้น
- MLP residual (E1.5): PyTorch + CUDA
- ทุก run ต้องบันทึก device, library version และ GPU model
- เมื่อรายงาน training/inference time ให้ระบุ device เสมอ และห้ามสรุปความเร็วเทียบข้าม device โดยตรง เพราะ GBR รันบน CPU ขณะที่ XGBoost/LightGBM/MLP อาจรันบน GPU
- ความแม่นยำ (RMSE/MAE ฯลฯ) เทียบข้าม device ได้ตามปกติ ตราบใดที่ split/feature/seed เท่ากัน

---

## 4. การแบ่งข้อมูลและการตรวจสอบแบบเวลา

### 4.1 Holdout หลัก

```text
Train                     Validation             Test (locked)
ก่อน validation_start     ถึงก่อน test_start     ตั้งแต่ test_start
```

- Train ใช้ฝึกโมเดล
- Validation ใช้เลือก feature, strategy, algorithm และ hyperparameter
- Test เปิดใช้ครั้งเดียวหลังเลือก final configuration

### 4.2 Walk-forward validation

ภายใน Train + Validation ให้สร้าง expanding-window folds เช่น 4 folds

```text
Fold 1: Train ──────► Validate
Fold 2: Train ───────────► Validate
Fold 3: Train ─────────────────► Validate
Fold 4: Train ───────────────────────► Validate
```

ข้อกำหนด:

- validation window ของทุก fold ต้องอยู่หลัง training window เสมอ
- ทุกสถานีใช้ขอบเขตวันที่เดียวกัน
- preprocessing ที่ต้อง fit เช่น imputer, encoder หรือ scaler ต้อง fit ใหม่ภายใน train ของแต่ละ fold
- ห้ามสุ่ม `KFold` และห้าม shuffle ข้อมูลข้ามเวลา

---

## 5. Baseline กลาง

กำหนด baseline ที่ใช้เป็นจุดอ้างอิงของ Stage 1

```yaml
training_strategy: local_station
algorithm: xgboost
forecast_strategy: direct
spatial_strategy: no_neighbor
target_transform: raw
hyperparameters: fixed_reasonable_defaults
```

Baseline นี้หมายถึง

- ฝึกโมเดลแยกสถานี
- ใช้ XGBoost
- สร้าง 7 โมเดลต่อสถานีสำหรับ `t+1` ถึง `t+7`
- ไม่ใช้ PM2.5 ของสถานีข้างเคียง
- ยังไม่ทำ Optuna ในขั้น screening

---

# 6. Experiment 1 — Training Strategy

## คำถามวิจัย

การรวมข้อมูลข้ามพื้นที่ในระดับใดให้ความแม่นยำและความสามารถในการ generalize ดีที่สุด

ระหว่าง E1 ให้ตรึง `algorithm=XGBoost`, `forecast=Direct`, `spatial=No neighbor` และ fixed hyperparameters

## E1.1 Local Station Model

ฝึกหนึ่งชุดโมเดลต่อหนึ่งสถานี

```text
Station A → 7 horizon models
Station B → 7 horizon models
Station C → 7 horizon models
```

ข้อดีคือเรียนรู้รูปแบบเฉพาะสถานีได้ดี แต่มีข้อมูลฝึกต่อโมเดลน้อย ใช้เป็น baseline ของ E1

## E1.2 Global Model

รวมทุกสถานีเพื่อฝึกโมเดลร่วมกัน โดยเพิ่มข้อมูลระบุตำแหน่ง

```text
All stations + station_id + latitude + longitude + elevation
                         ↓
                    Global model
```

หากจำนวนวันที่ใช้ได้ของแต่ละสถานีต่างกันมาก ให้รายงานจำนวนตัวอย่างรายสถานี และพิจารณา sample weight เพื่อไม่ให้สถานีที่มีข้อมูลมากครอบงำผลการเรียนรู้

## E1.3 Global + Local Residual Correction

E1.3 เป็น residual correction แบบ tree (local corrector เป็น tree boosting) ไม่ควรเรียกว่า neural-network fine-tuning ให้ใช้ global model ร่วมกับ local residual model (ส่วนตัวแปร local corrector แบบ MLP อยู่ใน E1.5)

\[
e_{s,t,h}=y_{s,t,h}-\hat{y}^{global}_{s,t,h}
\]

\[
\hat{y}^{final}_{s,t,h}
=\hat{y}^{global}_{s,t,h}+\hat{e}^{local}_{s,t,h}
\]

ขั้นตอน:

1. ฝึก global model ด้วย training fold
2. สร้าง out-of-fold prediction สำหรับแถว train
3. คำนวณ residual จาก out-of-fold prediction เท่านั้น
4. ฝึก local correction model ของแต่ละสถานีให้ทำนาย residual
5. validation/test ใช้ global prediction บวก local residual prediction

ห้ามใช้ in-sample prediction ของ global model เป็น target residual โดยตรง เพราะ residual จะดูง่ายเกินจริงและทำให้ผลประเมิน optimistic

## E1.4 Regional Model

แบ่งสถานีตามภูมิภาคที่กำหนดล่วงหน้า แล้วฝึกหนึ่งชุดโมเดลต่อภูมิภาค

```text
North stations      → Regional model: North
Central stations    → Regional model: Central
Northeast stations  → Regional model: Northeast
South stations      → Regional model: South
```

กติกา:

- region mapping ต้องกำหนดจากข้อมูลภูมิศาสตร์ ไม่ใช้ผล validation เป็นตัวแบ่ง
- ภูมิภาคที่มีสถานีน้อยหรือจำนวนข้อมูลไม่พอต้องกำหนด minimum sample rule ก่อนทดลอง
- หากสถานีอยู่นอกกลุ่ม ต้องกำหนด fallback เป็น Global model

## E1.5 Global XGBoost + Local MLP Residual

ใช้แนวคิด residual correction แบบเดียวกับ E1.3 แต่เปลี่ยน local corrector จาก tree
เป็น Multi-Layer Perceptron (MLP) เพื่อทดสอบว่า neural residual จับ pattern เฉพาะสถานี
ที่ global XGBoost พลาดได้ดีกว่าหรือไม่

\[
e_{s,t,h}=y_{s,t,h}-\hat{y}^{global}_{s,t,h}
\]

\[
\hat{y}^{final}_{s,t,h}=\hat{y}^{global}_{s,t,h}+\hat{e}^{MLP}_{s,t,h}
\]

ขั้นตอน:

1. ฝึก global XGBoost ด้วย training fold (ใช้ตัวเดียวกับ E1.2)
2. สร้าง out-of-fold prediction สำหรับแถว train
3. คำนวณ residual จาก out-of-fold prediction เท่านั้น
4. ฝึก MLP ของแต่ละสถานีให้ทำนาย residual
5. validation/test ใช้ global prediction บวก MLP residual prediction

Input ของ MLP:

- ใช้ feature ชุดเดียวกับ global model บวก `global_prediction` ของ horizon นั้นเป็น input เพิ่ม
- standardize numeric feature ทุกตัวด้วย scaler ที่ fit จาก train fold เท่านั้น
- categorical เข้ารหัสแบบเดียวกับ global model และ fit จาก train เท่านั้น

กติกากัน overfit (จำเป็นเพราะข้อมูลต่อสถานีน้อย):

- ฝึก MLP หนึ่งตัวต่อสถานีต่อ horizon
- ใช้ early stopping บน validation fold
- ใส่ dropout และ/หรือ L2 weight decay, กำหนด architecture เล็ก (เช่น 1–2 hidden layers)
- กำหนด minimum sample rule ต่อสถานี ถ้าไม่ถึงเกณฑ์ให้ fallback เป็น global prediction (residual=0) พร้อมสร้าง flag
- scaler, encoder และ MLP ทั้งหมด fit จาก train fold เท่านั้น
- ตั้ง seed และบันทึก architecture/hyperparameters ทุก run

Framework และ device:

- ใช้ PyTorch เป็น MLP framework (รองรับ CUDA); optional wrapper `skorch` เพื่อให้เข้ากับ walk-forward pipeline แบบ sklearn
- รันบน GPU เมื่อ `torch.cuda.is_available()` เป็นจริง มิฉะนั้น fallback เป็น CPU โดยผลลัพธ์เชิงตัวเลขต้องเทียบเคียงกันได้
- ย้ายทั้ง model และ tensor ไปยัง device เดียวกัน (`.to(device)`) และบันทึกว่า run นั้นใช้ device ใด
- ตั้ง `torch.manual_seed` และ `torch.cuda.manual_seed_all` เพื่อ reproducibility; ถ้าต้องการผลซ้ำแบบ deterministic ให้ตั้ง `torch.use_deterministic_algorithms(True)` และยอมรับว่าอาจช้าลง
- กำหนด batch size และใช้ mini-batch training บน GPU; ระบุ dtype (float32) ให้ชัด

ห้ามใช้ in-sample prediction ของ global model เป็น target residual โดยตรง เช่นเดียวกับ E1.3

## ผลลัพธ์ E1

เปรียบเทียบค่า macro-average ข้ามสถานีและ horizon รวมถึงรายงานรายสถานี เพื่อดูว่า strategy ที่ชนะโดยรวมทำให้บางสถานีแย่ลงหรือไม่

---

# 7. Experiment 2 — Algorithm

## คำถามวิจัย

เมื่อใช้ข้อมูลและ strategy เดียวกัน Algorithm ใดเหมาะกับการพยากรณ์ PM2.5 มากที่สุด

ระหว่าง E2 ให้ตรึง training, forecast และ spatial strategy ตาม baseline หรือผู้ชนะจาก E1 ตามลำดับแผนที่กำหนดไว้ก่อนเริ่มรัน

## E2.1 XGBoost

- objective เริ่มต้น: `reg:squarederror`
- ใช้ early stopping กับ validation fold
- รองรับ nonlinear interaction และ feature จำนวนมากได้ดี

## E2.2 LightGBM

- objective เริ่มต้น: regression/L2
- ควบคุม `num_leaves`, `max_depth`, `min_child_samples` เพื่อไม่ให้ overfit
- ใช้ early stopping และ validation folds เดียวกับ XGBoost

## E2.3 Gradient Boosting Regressor

- ใช้ `sklearn.ensemble.GradientBoostingRegressor`
- เป็น classical boosting baseline
- preprocessing และ evaluation cohort ต้องเหมือนสองโมเดลข้างต้น

## Fairness rule

Stage 1 ใช้ fixed reasonable defaults ที่มี model capacity ใกล้เคียงกัน ไม่ควรให้ algorithm หนึ่งผ่าน Optuna แล้วเปรียบเทียบกับค่า default ของอีก algorithm

Stage finalist ต้องได้รับ tuning budget เท่ากัน เช่น จำนวน trials, folds, timeout และ early-stopping policy เท่ากันเท่าที่ algorithm รองรับ

การเปรียบเทียบ timing (training/inference time) ต้องคำนึงถึง device เพราะ GBR รันบน CPU ขณะที่ XGBoost/LightGBM/MLP อาจรันบน GPU ให้เทียบความเร็วเฉพาะ algorithm ที่รันบน device ประเภทเดียวกัน หรือรายงานเวลาแยกตาม device เสมอ ส่วนความแม่นยำเทียบข้าม device ได้ตามปกติ

---

# 8. Experiment 3 — Forecasting Strategy

## คำถามวิจัย

การฝึกแยกตาม horizon หรือการพยากรณ์ทั้ง 7 วันในระบบเดียวให้ผลดีกว่ากัน

ในรายงานให้ใช้คำว่า **Direct single-output forecasting** และ **Multi-output forecasting** แทน single-head/multiple-head เพราะ tree model ไม่ได้มี shared neural head โดยตรง

## E3.1 Direct Single-output Forecasting

สร้างโมเดลแยก 7 ตัวต่อ training unit

```text
Model h1 → t+1
Model h2 → t+2
...
Model h7 → t+7
```

ข้อดี:

- แต่ละ horizon เรียนรูปแบบของตัวเอง
- tune hyperparameters แยก horizon ได้
- วิเคราะห์ feature importance/SHAP แยก horizon ได้ชัด

## E3.2 Multi-output Forecasting

ระบบเดียวรับ input แล้วให้ผล 7 ค่า

\[
\hat{\mathbf{y}}_t=
[\hat y_{t+1},\hat y_{t+2},...,\hat y_{t+7}]
\]

เพื่อความเท่าเทียมระหว่าง algorithm ให้ใช้ multi-output implementation ที่บันทึกไว้อย่างชัดเจน เช่น `MultiOutputRegressor` สำหรับโมเดลที่ไม่มี native shared multi-output และรายงานว่า wrapper ดังกล่าวยังประกอบด้วย estimator หลายตัวภายใน

## Fair comparison

- Direct และ Multi-output ต้องประเมินบนแถวที่มี target ครบทั้ง 7 horizons เหมือนกัน
- Feature set, split และ preprocessing เหมือนกัน
- รายงานทั้งเวลาฝึก ขนาดโมเดล และ inference time เพิ่มเติม เพราะความแม่นยำอาจใกล้กันแต่ต้นทุนต่างกัน

---

# 9. Experiment 4 — Spatial PM2.5

## คำถามวิจัย

ค่า PM2.5 จากสถานีอื่นช่วยพยากรณ์สถานีเป้าหมายหรือไม่ และการให้น้ำหนักเชิงกายภาพช่วยมากกว่าค่าเฉลี่ยธรรมดาหรือไม่

ให้ตรึง training strategy, algorithm และ forecast strategy ระหว่าง E4 ห้ามเปลี่ยนพร้อมกับ spatial features

## E4.1 No Neighbor

ใช้เฉพาะ PM2.5 lag ของสถานีเป้าหมายและ non-spatial baseline features

## E4.2 Unweighted Neighbor

สร้างค่าเฉลี่ย PM2.5 จากสถานีข้างเคียงที่กำหนดไว้ล่วงหน้า

\[
NeighborPM_{j,t}=\frac{1}{N_j}\sum_{i\in N_j}PM_{i,t}
\]

สร้าง lag อย่างน้อย 1, 3 และ 7 วันจากค่า aggregate นี้

## E4.3 Distance-weighted Neighbor

คำนวณระยะทางด้วยพิกัดแบบ Haversine แล้วใช้

\[
w^{dist}_{ij}=\frac{1}{d_{ij}+\epsilon}
\]

\[
NeighborPM^{dist}_{j,t}=
\frac{\sum_{i\in N_j}w^{dist}_{ij}PM_{i,t}}
{\sum_{i\in N_j}w^{dist}_{ij}}
\]

กำหนด `epsilon`, รัศมีสูงสุด และจำนวนเพื่อนบ้านสูงสุดใน config ก่อนดูผล validation

## E4.4 Wind + Distance-weighted Neighbor

ใช้ระยะทางร่วมกับความสอดคล้องของทิศทางลม

ให้ `bearing(j→i)` คือทิศจากสถานีเป้าหมายไปยังสถานีเพื่อนบ้าน และ `wind_from` คือทิศอุตุนิยมวิทยาที่ลมพัดมา หากเพื่อนบ้านอยู่ในทิศต้นลม ควรมีน้ำหนักสูง

\[
\Delta\theta_{ij,t}
=angular\_difference(wind\_from_{j,t},bearing(j\rightarrow i))
\]

\[
w^{wind}_{ij,t}=
\frac{\max(0,\cos(\Delta\theta_{ij,t}))^p}
{d_{ij}+\epsilon}
\]

\[
NeighborPM^{wind}_{j,t}=
\frac{\sum_i w^{wind}_{ij,t}PM_{i,t}}
{\sum_i w^{wind}_{ij,t}}
\]

กติกาเพิ่มเติม:

- ยืนยัน convention ของทิศลมว่าเป็น “พัดมาจาก” ไม่ใช่ “พัดไปยัง”
- แปลง degree/radian ให้ถูกต้องและมี unit test สำหรับทิศหลักทั้ง 4
- ถ้าลมสงบหรือผลรวมน้ำหนักเป็นศูนย์ ให้ fallback เป็น distance-weighted value พร้อมสร้าง flag
- PM2.5 ของเพื่อนบ้านต้องผ่าน lag ก่อนใช้งาน ห้ามใช้ค่าของวันอนาคต
- ถ้าสถานีข้างเคียงไม่มีข้อมูลในวันนั้น ให้ normalize น้ำหนักใหม่จากสถานีที่มีข้อมูล ห้ามเติมจากอนาคต

## Spatial sensitivity analysis

สำหรับ finalist ให้ทดสอบความไวอย่างจำกัด เช่น

- รัศมีเพื่อนบ้าน: 50, 100 และ 200 km
- จำนวนเพื่อนบ้าน: 3, 5 และ 10 สถานี
- distance decay: `1/d` เทียบกับ `1/d²`

เลือกจาก validation เท่านั้น และไม่รวมทุกค่ากับ full factorial ตั้งแต่ Stage 1

---

# 10. การออกแบบการ Cross Experiment

ไม่ทำ full factorial ทุกตัวเลือกตั้งแต่เริ่ม เพราะมีอย่างน้อย

\[
4\ training\ strategies
\times 3\ algorithms
\times 2\ forecast\ strategies
\times 4\ spatial\ strategies
=96\ configurations
\]

และหนึ่ง configuration อาจต้องฝึกหลายสถานี หลาย horizon และหลาย walk-forward folds

## Stage 0 — Data Audit และ Baseline Lock

1. ตรวจ duplicate, missingness, date coverage และจำนวนข้อมูลรายสถานี
2. ตรวจ feature leakage ด้วย feature timestamp audit
3. ตรึง dataset hash, split, feature baseline, metrics และ seed
4. รัน persistence/naive baseline และ XGBoost baseline

Naive baseline ที่ควรมี:

\[
\hat y_{t+h}=PM2.5_t
\]

โมเดล machine learning ต้องแสดงให้เห็นว่าเหนือกว่า naive baseline โดยเฉพาะ horizon ใกล้

## Stage 1 — One-factor-at-a-time Screening

เริ่มจาก baseline แล้วเปลี่ยนทีละปัจจัย

| กลุ่ม | Configurations |
|---|---|
| E1 | Local / Global / Global+Local tree residual / Global+Local MLP residual / Regional |
| E2 | XGBoost / LightGBM / GBR |
| E3 | Direct / Multi-output |
| E4 | No neighbor / Unweighted / Distance / Wind+Distance |

เมื่อหัก baseline ที่ซ้ำกัน จะมีประมาณ 11 configurations หลักใน screening แต่จำนวน model fits จริงขึ้นกับจำนวนสถานี, horizon และ folds

Stage 1 ใช้ fixed parameters และ walk-forward folds ชุดเดียวกัน จุดประสงค์คือคัดตัวเลือก ไม่ใช่ประกาศผู้ชนะสุดท้าย

## Stage 2 — Cross เฉพาะ Finalists

เลือกไม่เกิน 2 ตัวเลือกที่ดีสุดของแต่ละแกนโดยอิง primary metric และ stability จาก Stage 1

ตัวอย่าง:

```text
Training: Local, Global
Algorithm: XGBoost, LightGBM
Forecast: Direct, Multi-output
Spatial: No neighbor, Wind+Distance
```

จะได้

\[
2\times2\times2\times2=16\ configurations
\]

Stage 2 ใช้ตรวจ interaction เช่น spatial feature อาจช่วย Global model แต่ไม่ช่วย Local model

หาก compute จำกัด ให้เริ่มจาก pairwise crosses ต่อไปนี้ก่อน

1. Training × Spatial
2. Algorithm × Forecast
3. Training × Algorithm
4. Forecast × Spatial

## Stage 3 — Hyperparameter Optimization

ทำ Optuna เฉพาะ Top 4–8 configurations จาก Stage 2

- initial tuning: 50–100 trials/configuration
- final candidates: ขยายถึง 100–300 trialsเมื่อ improvement ยังไม่อิ่มตัวและ compute เพียงพอ
- objective: mean walk-forward macro RMSE
- early stopping ภายในแต่ละ fold
- บันทึก failed/pruned trials
- ห้ามใช้ Test metric เป็น Optuna objective

## Stage 4 — Final Model Selection

เลือก 1 primary model และอาจเก็บ 1 runner-up โดยใช้

1. Mean validation macro RMSE
2. Worst-station RMSE และ stability across folds
3. MAE ในวัน PM2.5 สูง
4. ความซับซ้อน เวลาฝึก และเวลา inference

## Stage 5 — Locked Final Test

1. Fix configuration และ hyperparameters ก่อนเปิด test
2. Retrain ด้วย Train + Validation โดยไม่เปลี่ยน pipeline
3. ประเมิน Test เพียงครั้งเดียว
4. รายงานทุก metric แยก horizon และ station
5. หากปรับโมเดลหลังเห็น Test ต้องระบุว่า test ไม่ได้เป็น untouched holdout อีกต่อไป และต้องมี test ชุดใหม่

---

# 11. Hyperparameter Search Space

## 11.1 XGBoost

```python
params = {
    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
    "max_depth": trial.suggest_int("max_depth", 3, 10),
    "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
    "n_estimators": 5000,
    "device": "cuda",
    "tree_method": "hist",
}
```

ใช้ `early_stopping_rounds=100` และ tune `learning_rate` คู่กับจำนวนรอบจริงที่ early stopping เลือก

`device="cuda"` และ `tree_method="hist"` เป็น fixed setting (ไม่ tune) สำหรับ XGBoost >= 2.0 หาก GPU ไม่พร้อมใช้งานให้ fallback เป็น `device="cpu"` โดยไม่เปลี่ยน search space อื่น

## 11.2 LightGBM

```python
params = {
    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
    "num_leaves": trial.suggest_int("num_leaves", 15, 255),
    "max_depth": trial.suggest_int("max_depth", 3, 12),
    "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
    "n_estimators": 5000,
}
```

ต้องตรวจความสัมพันธ์ระหว่าง `num_leaves` กับ `max_depth` เพื่อป้องกัน tree ที่ซับซ้อนเกินไป

optional GPU: ตั้ง `device_type="gpu"` (หรือ CUDA build) ได้ถ้า environment รองรับ มิฉะนั้นใช้ CPU โดยไม่เปลี่ยน search space อื่น

## 11.3 Gradient Boosting Regressor

```python
params = {
    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
    "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
    "max_depth": trial.suggest_int("max_depth", 2, 8),
    "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    "max_features": trial.suggest_float("max_features", 0.5, 1.0),
}
```

## 11.4 MLP Residual (PyTorch)

- hidden_layers: 1–2 ชั้น, hidden_units เช่น {32, 64, 128}
- dropout: 0.0–0.5
- weight_decay (L2): log range เช่น 1e-6 ถึง 1e-2
- learning_rate: log range เช่น 1e-4 ถึง 1e-2 (Adam)
- batch_size: {32, 64, 128}
- early stopping บน validation fold, max epochs พร้อม patience
- activation: ReLU (ค่าเริ่มต้น)
- device: CUDA เมื่อพร้อมใช้งาน มิฉะนั้น CPU

---

# 12. Objective และ Target Transformation เพิ่มเติม

ส่วนนี้เป็น secondary experiment หลังได้ finalists แล้ว ไม่ควรผสมใน Stage 1

## E5.1 Loss/Objective

- squared error
- absolute error
- pseudo-Huber เมื่อ algorithm รองรับ

## E5.2 Target

เปรียบเทียบ

```text
Raw target:    y = PM2.5
Log target:    y' = log1p(PM2.5)
```

กรณี log target ต้องแปลงกลับ

```python
y_pred_pm25 = np.expm1(y_pred_log)
```

และคำนวณ metric บนหน่วย µg/m³ เท่านั้น การ transform อาจช่วยข้อมูลที่เบ้ แต่ต้องตรวจว่าเหตุการณ์ PM2.5 สูงถูก underestimate หรือไม่

---

# 13. Metrics และหลักการจัดอันดับ

คำนวณอย่างน้อย

\[
MAE=\frac{1}{n}\sum|y_i-\hat y_i|
\]

\[
MSE=\frac{1}{n}\sum(y_i-\hat y_i)^2
\]

\[
RMSE=\sqrt{MSE}
\]

\[
R^2=1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}
\]

\[
Bias=\frac{1}{n}\sum(\hat y_i-y_i)
\]

## 13.1 ระดับการรายงาน

1. ราย horizon: `t+1` ถึง `t+7`
2. ราย station
3. station × horizon
4. micro-average จากทุกแถว
5. macro-average โดยเฉลี่ยแต่ละสถานีเท่ากัน
6. ค่าเฉลี่ยและส่วนเบี่ยงเบนข้าม walk-forward folds

Primary metric:

```text
Macro RMSE averaged across stations and horizons
```

Secondary metrics:

- Macro MAE
- R² และ Bias
- Worst-station RMSE
- RMSE degradation จาก t+1 ไป t+7
- training time, peak memory, model size และ inference time

## 13.2 High-PM2.5 Episode Metrics

รายงาน MAE/RMSE เฉพาะวันที่ค่าจริงสูงกว่า threshold ที่กำหนด เช่น

- PM2.5 > 37.5 µg/m³
- PM2.5 > 75 µg/m³

threshold ต้องกำหนดก่อนดูผลและอธิบายแหล่งอ้างอิงในรายงาน นอกจากนี้ให้รายงาน recall ของเหตุการณ์สูง หากมีการแปลงผลพยากรณ์เป็นการแจ้งเตือน

## 13.3 Statistical uncertainty

สำหรับ finalists ให้สร้าง 95% confidence interval ด้วย paired block bootstrap ตามช่วงเวลา หรือเปรียบเทียบ error แบบ paired บนวัน/สถานีเดียวกัน หลีกเลี่ยง bootstrap รายแถวแบบสุ่มอิสระเพราะข้อมูลอนุกรมเวลามี autocorrelation

---

# 14. Feature Ablation และ Explainability

หลังเลือก training/algorithm/forecast strategy แล้ว ให้ทำ ablation แบบเพิ่มทีละกลุ่ม

| Ablation | Feature group |
|---|---|
| A1 | PM2.5 lag only |
| A2 | + Weather |
| A3 | + Rolling/trend |
| A4 | + Calendar |
| A5 | + CO/AOD/hotspot |
| A6 | + Spatial PM2.5 |

ใช้ validation folds เดิมและไม่ tune ใหม่เต็มรูปแบบทุก ablation เว้นแต่มีเหตุผลชัดเจน

สำหรับ final model รายงาน

- gain importance
- permutation importance บน validation/test ที่ไม่ใช้ฝึก
- SHAP แยกอย่างน้อย t+1, t+3 และ t+7
- เปรียบเทียบว่า feature สำคัญเปลี่ยนตาม horizon หรือไม่

SHAP ใช้อธิบายพฤติกรรมของโมเดล ไม่ถือเป็นหลักฐานเชิงสาเหตุ

---

# 15. กฎป้องกัน Data Leakage

1. แบ่งข้อมูลตามเวลาก่อน fit preprocessing ทุกชนิด
2. สร้าง lag/rolling ภายใน station และใช้เฉพาะอดีต
3. ห้าม interpolate ข้ามจาก validation/test ย้อนเข้า train
4. ห้ามใช้ target ของเพื่อนบ้านในวันอนาคต แม้สถานีนั้นจะมีข้อมูลครบ
5. Global-local residual (ทั้ง tree และ MLP) ต้องใช้ out-of-fold global prediction ใน training residual และ scaler/encoder ของ MLP ต้อง fit จาก train fold เท่านั้น
6. Encoder, imputer, scaler และ feature selector fit จาก training fold เท่านั้น
7. เลือก station radius, feature group, threshold และ hyperparameter จาก validation เท่านั้น
8. ห้ามดู test leaderboard ระหว่างพัฒนา
9. เก็บ timestamp availability ของ AOD/ข้อมูลภายนอก หากข้อมูลเผยแพร่ล่าช้า ต้อง lag ตามเวลาที่ใช้งานจริง
10. ตรวจ sample overlap ของ target windows โดยเฉพาะเมื่อทำ cross-validation และใช้ gap/embargo ตามความเหมาะสมกับ horizon สูงสุด 7 วัน

แนะนำให้เว้น gap อย่างน้อย 7 วันระหว่างปลาย training label window กับต้น validation prediction origin หาก pipeline มีความเสี่ยงที่ target windows จะซ้อนกัน

---

# 16. Run Naming และ Artifact ที่ต้องบันทึก

ใช้ชื่อ run ที่อ่านได้และไม่ซ้ำ เช่น

```text
E1_GLOBAL__XGB__DIRECT__NO_NEIGHBOR__SEED42
E1_GLOBAL_XGB__LOCAL_MLP_RESIDUAL__DIRECT__NO_NEIGHBOR__SEED42
E4_LOCAL__XGB__DIRECT__WIND_DISTANCE__SEED42
S2_GLOBAL__LGBM__MULTI__WIND_DISTANCE__SEED42
```

แต่ละ run ต้องบันทึก

```text
artifacts/<run_id>/
├── config.yaml
├── dataset_manifest.json
├── fold_manifest.csv
├── metrics_overall.json
├── metrics_by_horizon.csv
├── metrics_by_station.csv
├── metrics_station_horizon.csv
├── predictions_validation.parquet
├── feature_list.txt
├── feature_importance.csv
├── training_log.txt
├── model/
└── figures/
```

`dataset_manifest.json` ควรมี dataset path/version, hash, จำนวนแถว, date range, station list, target coverage และ code commit hash

นอกจากนี้ `dataset_manifest.json`/`config.yaml` ต้องเก็บ field `device`, `torch_version`, `cuda_version`, `gpu_model` และ library versions ของทุก algorithm (XGBoost/LightGBM/GBR/PyTorch) ที่ใช้ในแต่ละ run

---

# 17. ตารางสรุปผลมาตรฐาน

## 17.1 Overall leaderboard

| Rank | Run ID | Training | Model | Forecast | Spatial | Macro RMSE | Macro MAE | R² | High-PM MAE | Time |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## 17.2 Results by horizon

| Run ID | Metric | t+1 | t+2 | t+3 | t+4 | t+5 | t+6 | t+7 | Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ... | RMSE | ... | ... | ... | ... | ... | ... | ... | ... |

## 17.3 Results by station

| Run ID | Station | N | RMSE | MAE | R² | Bias | High-PM MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| ... | ... | ... | ... | ... | ... | ... | ... |

## 17.4 Interaction table

| Training | Algorithm | Forecast | Spatial | Validation RMSE | Δ vs Baseline |
|---|---|---|---|---:|---:|
| Local | XGB | Direct | None | ... | 0.000 |
| Global | XGB | Direct | None | ... | ... |
| Global | XGB | Direct | Wind+Distance | ... | ... |

---

# 18. เกณฑ์เลือกโมเดลสุดท้าย

โมเดลสุดท้ายไม่ควรเลือกจาก RMSE ต่ำสุดเพียงหลักทศนิยมเล็กน้อย ให้พิจารณาตามลำดับ

1. Macro RMSE validation ดีและ confidence interval ไม่แสดงความด้อยชัดเจน
2. ไม่ทำให้สถานีใดสถานีหนึ่งแย่ลงรุนแรง
3. ผลคงที่ข้าม folds และ seeds
4. ทำได้ดีในวัน PM2.5 สูง
5. Bias ไม่สูงจน underestimate pollution episode อย่างต่อเนื่อง
6. ต้นทุน training/inference เหมาะกับการนำไปใช้
7. pipeline ทำซ้ำได้และไม่มี leakage warning

หากสองโมเดลใกล้เคียงกัน ให้เลือกโมเดลที่ง่ายกว่า เร็วกว่า และอธิบายได้ง่ายกว่าเป็น primary model

---

# 19. ลำดับดำเนินงานฉบับสรุป

```text
Data audit + leakage tests
        ↓
Lock dataset / split / features / metrics
        ↓
Naive + XGBoost baseline
        ↓
Stage 1: E1–E4 screening แบบเปลี่ยนทีละปัจจัย
        ↓
เลือก Top 2 ของแต่ละแกน
        ↓
Stage 2: Cross finalists เพื่อตรวจ interaction
        ↓
Stage 3: Optuna + walk-forward CV เฉพาะ Top 4–8
        ↓
Objective / target transform / feature ablation
        ↓
เลือก final configuration จาก validation
        ↓
Retrain ด้วย Train + Validation
        ↓
Locked Test หนึ่งครั้ง
        ↓
Metrics + High-PM analysis + SHAP + รายงานผล
```

---

# 20. Checklist ก่อนประกาศผล

- [ ] Dataset hash และ split dates ถูกบันทึกแล้ว
- [ ] ทุก feature ผ่าน timestamp/leakage audit
- [ ] ใช้ evaluation cohort เดียวกันในการเปรียบเทียบ
- [ ] มี naive baseline
- [ ] Stage 1 เปลี่ยนเพียงหนึ่งปัจจัยต่อครั้ง
- [ ] Stage 2 ตรวจ interaction ของ finalists
- [ ] Tuning budget ของ algorithm finalists เท่าเทียมกัน
- [ ] Test set ไม่เคยใช้เลือก feature/config/hyperparameter
- [ ] รายงาน MAE, MSE, RMSE, R² และ Bias
- [ ] รายงานแยก station และ t+1 ถึง t+7
- [ ] รายงานผลของวันที่ PM2.5 สูง
- [ ] Global-local residual (tree และ MLP) ใช้ out-of-fold predictions
- [ ] MLP residual: scaler/encoder fit จาก train เท่านั้น, มี early stopping และ fallback rule เมื่อข้อมูลสถานีไม่พอ
- [ ] บันทึก device, library version (PyTorch/CUDA/XGBoost/LightGBM) และ GPU model ในแต่ละ run
- [ ] Spatial wind direction มี unit tests
- [ ] มี config, predictions, logs และ environment versions ครบ
- [ ] Final result ทำซ้ำได้จากคำสั่งหรือ pipeline เดียว

เอกสารนี้เป็น experiment protocol กลาง หากต้องเปลี่ยนกติกาหลังเริ่มทดลอง ต้องเพิ่ม version และบันทึกเหตุผล เพื่อป้องกันการเลือกวิธีตามผลที่เห็นภายหลัง
