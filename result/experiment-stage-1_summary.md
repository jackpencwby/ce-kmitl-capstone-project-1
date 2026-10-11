# สรุปผล Experiment Stage 1 (E1–E4)

## สรุปสำหรับอ่านเร็ว

- รัน E1_1–E1_5 ใหม่ด้วย XGBoost recipe แบบ fixed-common ชุดเดียวกัน โดยไม่ใช้ best parameters จาก Global model ทุกวิธีใช้ source dataset hash `c247a1670b8345b644296e1408bfa7ce2072826f1374582446d2da17d7b7d4a2`, 218 สถานี และ split เดียวกัน
- เลือกวิธีฝึก E1 จาก validation primary macro RMSE: **E1_4 Regional ต่ำสุดที่ 9.3894** ตามด้วย E1_1 9.7963, E1_2 9.8845, E1_3 10.3026 และ E1_5 13.3344
- E1_2 มี test primary macro RMSE ต่ำสุดใน E1 ที่ **4.6373** แต่ใช้ test เพื่อรายงานผลเท่านั้น ไม่ใช้เลือกหรือปรับวิธีฝึก
- E1_5 ได้ validation/test 13.3344/7.0248 สูงกว่า persistence baseline 10.9931/5.6568 ทั้งสอง split
- E2–E4 เป็นการทดลองคนละปัจจัย และบางกลุ่มมี configuration หรือ dataset คนละชุด จึงรายงานแยกกลุ่ม ไม่จัดอันดับรวมกับ E1

## ผลหลัก: Primary macro RMSE

ค่าต่ำกว่าดีกว่า; `n` คือจำนวน prediction-target pairs ใน micro metrics ของ split นั้น

| ทดลอง | วิธี/ปัจจัย | Validation RMSE (n) | Test RMSE (n) |
|---|---|---:|---:|
| [E1_1](../experiment-stage-1/artifacts/E1_1_LOCAL__XGB_FIXED_COMMON__DIRECT__NO_NEIGHBOR__SEED42__20261003_133331/) | Local XGBoost | 9.7963 (205,287) | 5.4830 (229,588) |
| [E1_2](../experiment-stage-1/artifacts/E1_2_GLOBAL__XGB_FIXED_COMMON__DIRECT__NO_NEIGHBOR__SEED42__20261003_134217/) | Global XGBoost | 9.8845 (205,287) | **4.6373 (229,588)** |
| [E1_3](../experiment-stage-1/artifacts/E1_3_GLOBAL_XGB_FIXED_COMMON__LOCAL_TREE_RESIDUAL__DIRECT__NO_NEIGHBOR__SEED42__20261003_140236/) | Global + local tree residual | 10.3026 (205,287) | 5.0538 (229,588) |
| [E1_4](../experiment-stage-1/artifacts/E1_4_REGIONAL__XGB_FIXED_COMMON__DIRECT__NO_NEIGHBOR__SEED42__20261003_140515/) | Regional XGBoost | **9.3894 (205,287)** | 4.9038 (229,588) |
| [E1_5](../experiment-stage-1/artifacts/E1_GLOBAL_XGB__LOCAL_MLP_RESIDUAL__DIRECT__NO_NEIGHBOR__SEED42__20261003_143629/) | Global + local MLP residual | 13.3344 (205,287) | 7.0248 (229,588) |
| [E2_1](./experiment-stage-1_artifacts_E2/E2_1/E2_1_metrics_overall.json) | XGBoost | 10.0661 (205,287) | 4.6381 (229,588) |
| [E2_2](./experiment-stage-1_artifacts_E2/E2_2/E2_2_metrics_overall.json) | LightGBM | 10.0840 (205,287) | 4.5967 (229,588) |
| [E2_3](./experiment-stage-1_artifacts_E2/E2_3/E2_3_metrics_overall.json) | GradientBoostingRegressor | 10.6336 (205,287) | 4.8943 (229,588) |
| [E3_1](./experiment-stage-1_artifacts_E3/E3_1/E3_1_metrics_overall.json) | Direct forecast | 10.4573 (199,339) | **5.8148 (215,257)** |
| [E3_2](./experiment-stage-1_artifacts_E3/E3_2/E3_2_metrics_overall.json) | Multi-output forecast | **10.4537 (199,339)** | 5.8586 (215,257) |
| [E4_1](./experiment-stage-1_artifacts_E4/E4_1/E4_1_metrics_overall.json) | ไม่มี neighbor features | **10.4762 (205,287)** | 5.7604 (229,588) |
| [E4_2](./experiment-stage-1_artifacts_E4/E4_2/E4_2_metrics_overall.json) | Neighbor features ไม่ถ่วงน้ำหนัก | 10.4786 (205,287) | 5.7755 (229,588) |
| [E4_3](./experiment-stage-1_artifacts_E4/E4_3/E4_3_metrics_overall.json) | Neighbor ถ่วงน้ำหนักตามระยะทาง | 10.4814 (205,287) | **5.7593 (229,588)** |
| [E4_4](./experiment-stage-1_artifacts_E4/E4_4/E4_4_metrics_overall.json) | Neighbor ถ่วงน้ำหนักตามลมและระยะทาง | 10.4963 (205,287) | 5.7682 (229,588) |

## ผลแยกตามกลุ่ม

### E1 — วิธีฝึก

ใช้ XGBoost parameters ร่วมกัน: `learning_rate=0.05`, `max_depth=5`, `min_child_weight=20`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0`, `reg_lambda=1`, `gamma=0`, `n_estimators=2000` สูงสุด และ early stopping 100 รอบจาก validation. ใช้ข้อมูล 192,768 แถว: train 125,828, validation 31,680, test 35,260; validation เริ่ม `2025-11-19` และ test เริ่ม `2026-04-13`.

E1_4 เป็นตัวเลือกนำตาม validation. E1_2 ได้ test ต่ำสุดในกลุ่ม แต่ test เป็น holdout สำหรับรายงาน ไม่ใช่เกณฑ์เลือก. ผลนี้มาจาก seed 42 เพียง seed เดียว จึงยังไม่แสดงความแปรปรวนระหว่าง seeds. E1_3 และ E1_5 สร้าง out-of-fold predictions ภายใน train เพื่อสร้าง residual targets; outer walk-forward folds เป็นศูนย์ตาม E1 holdout policy.

### E2 — อัลกอริทึม

E2_1 เป็น validation leader ในกลุ่มที่ 10.0661; E2_2 มี test ต่ำสุดในกลุ่มที่ 4.5967. E2 ใช้ configuration ของการทดลอง E2 ไม่ใช่ fixed-common recipe แบบ E1. LightGBM ใช้ CPU ใน environment นี้.

### E3 — รูปแบบการพยากรณ์

E3_2 มี validation ต่ำกว่า E3_1 เล็กน้อย (10.4537 เทียบกับ 10.4573); E3_1 มี test ต่ำกว่าเล็กน้อย (5.8148 เทียบกับ 5.8586). ทั้งคู่มี test pairs 215,257 จึงควรพิจารณา support เมื่อนำไปเทียบกับกลุ่มที่มี 229,588 pairs.

### E4 — ข้อมูลจากสถานีข้างเคียง

Validation อยู่ในช่วง 10.4762–10.4963 และ test อยู่ในช่วง 5.7593–5.7755. คะแนนใกล้กันเมื่อเพิ่ม neighbor features; E4_1 นำบน validation ขณะที่ E4_3 นำบน test เล็กน้อย. E4 ใช้ dataset hash `7e160e7b…` ซึ่งเป็น dataset revision คนละชุดกับ E1–E3.

## แนวทางเลือกโมเดล

เลือก configuration จาก primary validation metric ภายในคำถามทดลองและ cohort เดียวกัน แล้วรายงาน test แยกเป็นผล holdout. สำหรับคำถาม E1 เรื่องวิธีฝึก ให้ใช้ E1_4 เป็นตัวเลือกตาม validation ในรอบนี้ และเก็บ E1_2 test score เป็นผลประกอบ. ควรรันหลาย seeds หากต้องการยืนยันความคงทนของความต่าง.

## แหล่งข้อมูลและข้อจำกัด

E1 ใช้ artifacts ของ fixed-common runs วันที่ 2026-10-03 ใน `experiment-stage-1/artifacts/`; รายละเอียด metrics อยู่ใน [experiment-stage-1_metric_comparison.md](./experiment-stage-1_metric_comparison.md). ผล E2–E4 อ้างอิง artifacts ที่ลิงก์ในตาราง. คะแนนเป็น holdout validation/test ไม่ได้นำ walk-forward CV มาเฉลี่ยรวม. E1 master SHA-256 คือ `c247a1670b8345b644296e1408bfa7ce2072826f1374582446d2da17d7b7d4a2`.
