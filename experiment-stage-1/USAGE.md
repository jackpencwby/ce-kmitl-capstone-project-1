# คู่มือการใช้งาน Stage 1 (ภาษาไทย)

คู่มือนี้สอนวิธีรันการทดลอง Stage 1 สำหรับโมเดลพยากรณ์ PM2.5 ล่วงหน้า 1–7 วัน
ตาม [`Experimental_Plan.md`](../Experimental_Plan.md) แบบทีละขั้นตอน
ตั้งแต่การติดตั้ง การเชื่อมต่อ GCP การรันแต่ละ Experiment ไปจนถึงการอ่านผลลัพธ์

> สรุปสั้น ๆ: การทดลองแต่ละอันคือไฟล์ `.py` แยกกัน (เช่น `E1_1.py`, `E2_2.py`)
> ทุกไฟล์มี flag `--train` และ `--stations` เหมือนกัน โค้ดส่วนกลางอยู่ในโฟลเดอร์
> `common/` เพื่อให้ทุกการทดลองใช้กติกา ข้อมูล และการวัดผลชุดเดียวกัน

---

## 1. สิ่งที่ต้องเตรียม

- Python 3.12 (ใช้ venv ของ repo ที่ `.venv` — เป็นตัวที่รองรับ CUDA อยู่แล้ว)
- ไฟล์ข้อมูลหลัก `clean-data_preprocess_all_stations_daily.csv` ที่ root ของ repo
  (มีอยู่แล้ว) หรือจะโหลดจาก GCS ก็ได้
- ไฟล์ `.env` ที่ root ของ repo (มี HMAC key ของ Cloud Storage อยู่แล้ว)

### ติดตั้ง dependency

รันจาก **root ของ repo** (โฟลเดอร์ `ce-kmitl-capstone-project-1`):

```powershell
.venv/Scripts/python.exe -m pip install -r experiment-stage-1/requirements.txt
```

จะติดตั้ง `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `torch`,
`boto3` (สำหรับต่อ GCS) และ `pyarrow` (สำหรับบันทึกผลเป็น parquet)

---

## 2. รูปแบบคำสั่งพื้นฐาน

ทุกคำสั่งรันจาก **root ของ repo** และใช้ python จาก `.venv`:

```powershell
.venv/Scripts/python.exe experiment-stage-1/<ไฟล์การทดลอง> [flags]
```

ตัวอย่างเร็วที่สุด (dry run — ยังไม่เทรน แค่ดูว่าจะเทรนอะไรบ้าง):

```powershell
.venv/Scripts/python.exe experiment-stage-1/E1_1.py
```

---

## 3. Flag ที่ใช้ได้ (เหมือนกันทุกไฟล์)

| Flag | ความหมาย |
|---|---|
| `--train` | **สั่งเทรนจริง** และบันทึก artifacts ถ้าไม่ใส่ = dry run (แค่พิมพ์ config + รายชื่อสถานี) |
| `--stations 72 36 ...` | เลือกสถานีที่จะเทรน/วัดผล ใส่ได้หลายตัว |
| `--stations all` | ใช้ทุกสถานีที่มีข้อมูล PM2.5 พอ (ค่าเริ่มต้น) |
| `--source auto` | อ่านไฟล์ local ถ้ามี ไม่มีค่อยโหลดจาก GCS (ค่าเริ่มต้น) |
| `--source local` | บังคับใช้ไฟล์ local เท่านั้น |
| `--source gcs` | บังคับโหลดใหม่จาก GCS |
| `--check-gcs` | ทดสอบการเชื่อมต่อ GCP จาก `.env` ก่อนรัน |
| `--max-stations N` | จำกัดจำนวนสถานี (ใช้ทดสอบเร็ว ๆ) |
| `--prefer-cpu` | บังคับใช้ CPU แม้มี GPU |
| `--upload-gcs` | หลังเทรนเสร็จ **อัปโหลดโฟลเดอร์ artifacts ทั้งหมดขึ้น GCS** (ต้องใช้คู่กับ `--train`) |
| `--gcs-artifacts-prefix PATH` | เปลี่ยนโฟลเดอร์ปลายทางใน bucket (ค่าเริ่มต้น `experiment-stage-1/artifacts`) |
| `--no-keep-local` | ลบโฟลเดอร์ artifacts บนเครื่องหลังอัปโหลดขึ้น GCS สำเร็จ (ค่าเริ่มต้นคือเก็บไว้) |
| `--log-level DEBUG` | ระดับ log (DEBUG / INFO / WARNING) |

> **flag ที่ตอบโจทย์ "จะเทรนกับสถานีไหน"** คือ `--stations`
> เช่น `--stations 72` เทรนสถานี 72 อย่างเดียว, `--stations 72 36 108` เทรน 3 สถานี,
> `--stations all` เทรนทุกสถานี

---

## 4. ตารางไฟล์การทดลองทั้งหมด

| ไฟล์ | ข้อ | สิ่งที่เปลี่ยน (ตัวแปรที่ทดลอง) | แกนที่ตรึงไว้ |
|---|---|---|---|
| `E1_1.py` | E1.1 | Training = **Local** (แยกโมเดลต่อสถานี) — baseline | XGB / Direct / ไม่ใช้เพื่อนบ้าน |
| `E1_2.py` | E1.2 | Training = **Global** (รวมทุกสถานี + station_id) | XGB / Direct / ไม่ใช้เพื่อนบ้าน |
| `E1_3.py` | E1.3 | Training = **Global + local tree residual** | XGB / Direct / ไม่ใช้เพื่อนบ้าน |
| `E1_4.py` | E1.4 | Training = **Regional** (แบ่งภาค เหนือ/อีสาน/กลาง/ใต้) | XGB / Direct / ไม่ใช้เพื่อนบ้าน |
| `E1_5.py` | E1.5 | Training = **Global + local MLP residual** (PyTorch) | XGB / Direct / ไม่ใช้เพื่อนบ้าน |
| `E2_1.py` | E2.1 | Algorithm = **XGBoost** | Local / Direct / ไม่ใช้เพื่อนบ้าน |
| `E2_2.py` | E2.2 | Algorithm = **LightGBM** (GPU ถ้ารองรับ ไม่งั้น CPU) | Local / Direct / ไม่ใช้เพื่อนบ้าน |
| `E2_3.py` | E2.3 | Algorithm = **GradientBoostingRegressor** (CPU) | Local / Direct / ไม่ใช้เพื่อนบ้าน |
| `E3_1.py` | E3.1 | Forecast = **Direct** (7 โมเดล แยก horizon) | Local / XGB / ไม่ใช้เพื่อนบ้าน |
| `E3_2.py` | E3.2 | Forecast = **Multi-output** (โมเดลเดียว 7 ค่า) | Local / XGB / ไม่ใช้เพื่อนบ้าน |
| `E4_1.py` | E4.1 | Spatial = **ไม่ใช้เพื่อนบ้าน** | Local / XGB / Direct |
| `E4_2.py` | E4.2 | Spatial = **เฉลี่ยเพื่อนบ้านแบบไม่ถ่วงน้ำหนัก** | Local / XGB / Direct |
| `E4_3.py` | E4.3 | Spatial = **ถ่วงน้ำหนักด้วยระยะทาง** 1/(d+ε) | Local / XGB / Direct |
| `E4_4.py` | E4.4 | Spatial = **ถ่วงน้ำหนักด้วยระยะทาง + ทิศลม** | Local / XGB / Direct |

---

## 5. ตัวอย่างการใช้งานทีละสถานการณ์

### 5.1 ดูก่อนว่าจะเทรนอะไร (dry run)

```powershell
.venv/Scripts/python.exe experiment-stage-1/E1_1.py --stations 72
```

จะพิมพ์ config, จำนวนสถานีที่เลือก, จำนวนแถว train/validation/test และจำนวน feature
โดย **ยังไม่เทรน** เหมาะกับการเช็คว่าตั้งค่าถูกก่อนรันจริง

### 5.2 เทรนจริงกับสถานีเดียว

```powershell
.venv/Scripts/python.exe experiment-stage-1/E1_1.py --train --stations 72
```

### 5.3 เทรนหลายสถานี

```powershell
.venv/Scripts/python.exe experiment-stage-1/E1_2.py --train --stations 72 36 108
```

### 5.4 เทรนทุกสถานี โดยดึงข้อมูลจาก GCS

```powershell
.venv/Scripts/python.exe experiment-stage-1/E1_2.py --train --stations all --source gcs
```

### 5.5 รันการทดลองเชิงพื้นที่ (ใช้ PM2.5 เพื่อนบ้าน + ทิศลม)

```powershell
.venv/Scripts/python.exe experiment-stage-1/E4_4.py --train --stations 72 36 108
```

### 5.6 ทดสอบการเชื่อมต่อ GCP อย่างเดียว

```powershell
.venv/Scripts/python.exe experiment-stage-1/E1_1.py --check-gcs
```

ถ้าสำเร็จจะขึ้น `GCS connection OK: gs://.../all_stations_daily.csv`

### 5.7 บังคับใช้ CPU (เช่น เครื่องไม่มี GPU)

```powershell
.venv/Scripts/python.exe experiment-stage-1/E1_5.py --train --stations 72 36 --prefer-cpu
```

---

## 6. การเชื่อมต่อ GCP (ผ่าน `.env`)

ระบบอ่านค่าจาก `.env` ที่ root ของ repo และต่อ Cloud Storage ผ่าน
endpoint แบบ S3-compatible (`https://storage.googleapis.com`) ด้วย `boto3`
โดยใช้คีย์ต่อไปนี้:

```
AWS_ACCESS_KEY_ID       = HMAC access key (ขึ้นต้นด้วย GOOG1E...)
AWS_SECRET_ACCESS_KEY   = HMAC secret
GCS_BUCKET              = kmitl-capstone-project-data-bucket
```

ไฟล์ข้อมูลหลักอยู่ที่
`gs://<GCS_BUCKET>/clean-data/preprocess/all_stations_daily.csv`
ระบบจะใช้ไฟล์ local ก่อนเสมอ ถ้าต้องการโหลดใหม่จาก GCS ให้ใช้ `--source gcs`

> ไม่ต้องแก้โค้ดใด ๆ แค่มี `.env` ที่ถูกต้องก็เชื่อมต่อได้

---

## 7. ผลลัพธ์ที่ได้ (Artifacts)

ทุกครั้งที่รันด้วย `--train` ระบบจะสร้างโฟลเดอร์ใหม่ที่
`experiment-stage-1/artifacts/<run_id>__<timestamp>/` ซึ่งประกอบด้วย:

| ไฟล์ | คืออะไร |
|---|---|
| `config.json` | ค่า config ทั้งหมด + device/GPU + เวอร์ชัน library + seed |
| `dataset_manifest.json` | path ข้อมูล, hash, จำนวนแถว, ช่วงวันที่, รายชื่อสถานี, git commit |
| `fold_manifest.csv` | ขอบเขตวันที่ของแต่ละ walk-forward fold |
| `metrics_overall.json` | สรุป micro/macro (มี primary macro RMSE) |
| `metrics_by_horizon.csv` | ผลแยกตาม horizon t+1..t+7 |
| `metrics_by_station.csv` | ผลแยกตามสถานี + episode วัน PM2.5 สูง |
| `metrics_station_horizon.csv` | ผลระดับ สถานี × horizon |
| `predictions_validation.parquet` | ค่าทำนายบน validation (ถ้าไม่มี pyarrow จะเป็น .csv) |
| `feature_list.txt` | รายชื่อ feature ที่ใช้ |
| `feature_importance.csv` | ความสำคัญของ feature |
| `training_log.txt` | log สรุปของ run |

### การอ่านผลบนหน้าจอ

หลังเทรนเสร็จ จะพิมพ์สรุปสั้น ๆ เช่น:

```
=== Validation results ===
Primary macro RMSE (station x horizon): 5.8626   <- ตัวชี้วัดหลัก (ยิ่งน้อยยิ่งดี)
Macro RMSE (station)                  : 5.8898
Macro MAE  (station)                  : 4.8644
Worst-station RMSE                    : 8.4624   <- สถานีที่แย่ที่สุด
Naive persistence primary macro RMSE  : 6.3907   <- baseline y(t+h)=y(t)
Elapsed seconds                       : 38.7
```

**ตัวชี้วัดหลัก (primary metric)** คือ `Primary macro RMSE` (เฉลี่ยข้าม
สถานี × horizon) โมเดลควรทำได้ **ดีกว่า** `Naive persistence` จึงจะถือว่ามีประโยชน์

### 7.1 อัปโหลด artifacts ขึ้น Cloud Storage

ถ้าอยากให้ผลลัพธ์ขึ้น GCS อัตโนมัติหลังเทรนเสร็จ ให้เติม `--upload-gcs`
(ใช้คู่กับ `--train` เสมอ) ระบบจะอัปโหลดทั้งโฟลเดอร์ run ขึ้นไปที่
`gs://<GCS_BUCKET>/experiment-stage-1/artifacts/<run_id>__<timestamp>/`
โดยเก็บโครงสร้างไฟล์เหมือนบนเครื่อง:

```powershell
.venv/Scripts/python.exe experiment-stage-1/E1_1.py --train --stations 72 --upload-gcs
```

หลังรันจะพิมพ์ทั้ง path บนเครื่องและ URI บน GCS:

```
Artifacts (local): ...\experiment-stage-1\artifacts\E1_LOCAL__...__20260920_162545
Artifacts (GCS) : gs://kmitl-capstone-project-data-bucket/experiment-stage-1/artifacts/E1_LOCAL__...__20260920_162545
```

- เปลี่ยนโฟลเดอร์ปลายทางใน bucket ได้ด้วย `--gcs-artifacts-prefix <path>`
- อยากให้ลบสำเนาบนเครื่องหลังอัปโหลดสำเร็จ ใช้ `--no-keep-local`
- ถ้าอัปโหลดล้มเหลว ระบบจะ **ไม่ลบ** ไฟล์บนเครื่อง และแจ้ง `[warn]` ให้ทราบ
  (run ไม่หาย เทรนใหม่ไม่ต้อง)
- ใช้ credential ชุดเดียวกับการอ่านข้อมูล (จาก `.env`) ไม่ต้องตั้งค่าเพิ่ม

---

## 8. ลำดับการรัน Stage 1 ที่แนะนำ

ตามแผน Stage 1 คือ "เปลี่ยนทีละปัจจัย" (one-factor-at-a-time):

1. **Baseline / E1** — รัน `E1_1` → `E1_5` เปรียบเทียบ training strategy
2. **E2** — รัน `E2_1` → `E2_3` เปรียบเทียบ algorithm
3. **E3** — รัน `E3_1`, `E3_2` เปรียบเทียบ direct vs multi-output
4. **E4** — รัน `E4_1` → `E4_4` เปรียบเทียบการใช้ PM2.5 เพื่อนบ้าน

ตัวอย่างรันทั้ง E1 กับสถานีชุดเดียวกัน:

```powershell
$stations = "72","36","108"
.venv/Scripts/python.exe experiment-stage-1/E1_1.py --train --stations $stations
.venv/Scripts/python.exe experiment-stage-1/E1_2.py --train --stations $stations
.venv/Scripts/python.exe experiment-stage-1/E1_3.py --train --stations $stations
.venv/Scripts/python.exe experiment-stage-1/E1_4.py --train --stations $stations
.venv/Scripts/python.exe experiment-stage-1/E1_5.py --train --stations $stations
```

จากนั้นเทียบค่า `Primary macro RMSE` ของแต่ละ run เพื่อคัดตัวเลือกที่ดีเข้าสู่ Stage 2

> **หมายเหตุ:** Stage 1 ใช้ "คัดตัวเลือก" ไม่ใช่ประกาศผู้ชนะสุดท้าย และ
> **ห้ามแตะชุด Test** จนกว่าจะเลือก final configuration เสร็จ (ตามแผนข้อ 3, 5)

---

## 9. รัน Unit Test

มี unit test สำหรับฟังก์ชันคำนวณระยะทาง/ทิศลม (ตามแผนข้อ 9 ที่กำหนดให้มี):

```powershell
.venv/Scripts/python.exe -m unittest discover -s experiment-stage-1/tests -v
```

---

## 10. ปัญหาที่พบบ่อย

- **`None of the requested stations are eligible pm25 targets`**
  สถานีที่ขอมาเป็นสถานีตรวจอากาศที่ไม่มีเซนเซอร์ PM2.5 (ค่า pm25 หายเกิน 30%)
  ระบบตัดออกอัตโนมัติ ลองใช้ id อื่น หรือดูรายชื่อสถานีที่ใช้ได้จาก dry run

- **LightGBM ขึ้น `GPU Tree Learner was not enabled`**
  เวอร์ชันที่ติดตั้งจาก pip เป็น CPU-only ระบบจะ fallback ไป CPU ให้อัตโนมัติ
  (ผลความแม่นยำเทียบกันได้ แต่ **อย่านำเวลา CPU ไปเทียบกับ GPU ตรง ๆ** ตามแผนข้อ 3.3)

- **E1_3 / E1_5 ให้ผลเท่ากันเมื่อใช้สถานีน้อย**
  เพราะข้อมูล residual ต่อสถานีไม่ถึงเกณฑ์ขั้นต่ำ ระบบจึง fallback เป็น residual = 0
  (ใช้ค่าจาก global อย่างเดียว) จะเห็นผลต่างชัดเมื่อใช้สถานีจำนวนมากขึ้น

- **ต้องรันจาก root ของ repo เสมอ** ไม่งั้น import `common` ไม่เจอ

---

ดูรายละเอียดการออกแบบ feature และเหตุผลเชิงวิชาการเพิ่มเติมได้ที่
[`../pm25-tft-model/FEATURE_SELECTION.md`](../pm25-tft-model/FEATURE_SELECTION.md)
และโปรโตคอลการทดลองฉบับเต็มที่ [`../Experimental_Plan.md`](../Experimental_Plan.md)
