# สรุปผล Experiment Stage 1 (E1–E4)

## สรุปสำหรับอ่านเร็ว

- ในกลุ่ม E1 ที่ใช้ dataset hash เดียวกัน (`E1_1`–`E1_3`) การฝึกแบบ Global XGBoost (`E1_2`) ให้ผลดีที่สุด: validation primary macro RMSE **9.713** และ test **4.618** ต่ำกว่า Local (`E1_1`) และ Global + local tree residual (`E1_3`).
- `E1_4` มี validation RMSE ต่ำสุดในผลทั้งหมดที่ **9.316** และ test RMSE **4.777** แต่ใช้ dataset hash คนละชุดกับ `E1_1`–`E1_3`; ควรยืนยันข้อมูลก่อนนำมาเทียบตรง ๆ
- E3 Direct กับ Multi-output ให้คะแนนใกล้กันมาก (validation **10.457** กับ **10.454**, test **5.815** กับ **5.859**). E4 ไม่พบประโยชน์ชัดเจนจาก neighbor features; คะแนนแทบไม่ต่างจากแบบไม่มีเพื่อนบ้าน
- Test ของ `E2_1` และ `E2_2` ซ้ำกับ validation ทั้งตัวชี้วัดและไฟล์ predictions จึงยังใช้จัดอันดับ test ไม่ได้ ส่วน `E2_3` มี test แยกตามปกติ

## ผลคะแนนหลัก

ใช้ **primary macro RMSE (station × horizon)** เป็นคะแนนหลัก; ค่ายิ่งต่ำยิ่งดี คะแนน validation และ test อ่านจาก `metrics_overall.json` และ `metrics_overall_test.json` ตามลำดับ `n` คือจำนวน prediction-target pairs ที่นำมาคิดคะแนน

| ทดลอง | ปัจจัยที่เปลี่ยน | Validation RMSE (n) | Test RMSE (n) |
|---|---|---:|---:|
| [E1.1](./experiment-stage-1_artifacts_E1/E1_1/E1_1_metrics_overall.json) | ฝึก Local | 9.861 (205,287) | 5.488 (229,588) |
| [E1.2](./experiment-stage-1_artifacts_E1/E1_2/E1_2_metrics_overall.json) | ฝึก Global | 9.713 (205,287) | **4.618 (229,588)** |
| [E1.3](./experiment-stage-1_artifacts_E1/E1_3/E1_3_metrics_overall.json) | Global + local tree residual | 10.233 (205,287) | 5.020 (229,588) |
| [E1.4](./experiment-stage-1_artifacts_E1/E1_4/E1_4_metrics_overall.json) | ฝึก Regional | **9.316 (205,287)** | 4.777 (229,588) |
| [E1.5](./experiment-stage-1_artifacts_E1/E1_5/E1_5_metrics_overall.json) | Global + local MLP residual | 13.695 (205,287) | 7.131 (229,588) |
| [E2.1](./experiment-stage-1_artifacts_E2/E2_1/E2_1_metrics_overall.json) | XGBoost | 10.611 (205,287) | 10.611* (205,287) |
| [E2.2](./experiment-stage-1_artifacts_E2/E2_2/E2_2_metrics_overall.json) | LightGBM | **10.609 (205,287)** | 10.609* (205,287) |
| [E2.3](./experiment-stage-1_artifacts_E2/E2_3/E2_3_metrics_overall.json) | GradientBoostingRegressor | 10.634 (205,287) | 4.894 (229,588) |
| [E3.1](./experiment-stage-1_artifacts_E3/E3_1/E3_1_metrics_overall.json) | Direct forecast | 10.457 (199,339) | **5.815 (215,257)** |
| [E3.2](./experiment-stage-1_artifacts_E3/E3_2/E3_2_metrics_overall.json) | Multi-output forecast | **10.454 (199,339)** | 5.859 (215,257) |
| [E4.1](./experiment-stage-1_artifacts_E4/E4_1/E4_1_metrics_overall.json) | ไม่มี neighbor features | **10.476 (205,287)** | 5.760 (229,588) |
| [E4.2](./experiment-stage-1_artifacts_E4/E4_2/E4_2_metrics_overall.json) | Neighbor แบบไม่ถ่วงน้ำหนัก | 10.479 (205,287) | 5.776 (229,588) |
| [E4.3](./experiment-stage-1_artifacts_E4/E4_3/E4_3_metrics_overall.json) | Neighbor ถ่วงตามระยะทาง | 10.481 (205,287) | **5.759 (229,588)** |
| [E4.4](./experiment-stage-1_artifacts_E4/E4_4/E4_4_metrics_overall.json) | Neighbor ถ่วงตามลมและระยะทาง | 10.496 (205,287) | 5.768 (229,588) |

ตัวหนาแสดงคะแนนต่ำสุดภายในแกนทดลองและ split เดียวกัน; ยังไม่ระบุผู้ชนะ test ของ E2 เพราะ artifacts สองชุดยืนยันไม่ได้

\* `E2_1` และ `E2_2` มีค่า test เหมือน validation ทุกค่า; `predictions_test.parquet` มี SHA-256 เหมือน `predictions_validation.parquet` ด้วย จึงควรถือ test ของสองรันนี้ว่า **ยังยืนยันไม่ได้** จนกว่าจะตรวจหรือรันใหม่

## สิ่งที่ผลทดลองบอก

### E1 — วิธีฝึก

ในชุดข้อมูลเดียวกันของ `E1_1`–`E1_3`, Global (`E1_2`) ดีกว่า Local และ Global + local tree residual ทั้ง validation และ test ส่วน MLP residual (`E1_5`) ได้คะแนนแย่สุดในกลุ่ม E1 อย่างชัดเจน ผล Regional (`E1_4`) ดูดีที่สุดด้าน validation และทำได้ดีบน test เช่นกัน แต่ต้องระวังว่าใช้ dataset คนละ hash กับ E1.1–E1.3

### E2 — อัลกอริทึม

คะแนน validation ของ XGBoost, LightGBM และ GBR ใกล้กันมาก: ช่วง **10.609–10.634** RMSE จึงยังไม่มีความต่างเด่นชัดจากคะแนนนี้เพียงอย่างเดียว Test ของ GBR คือ **4.894**; ยังสรุปผู้ชนะด้าน test ระหว่างสามอัลกอริทึมไม่ได้ เพราะ test artifacts ของ XGBoost และ LightGBM ซ้ำกับ validation

### E3 — รูปแบบการพยากรณ์

Direct มี test RMSE ต่ำกว่า Multi-output เล็กน้อย (**5.815 เทียบกับ 5.859**); Multi-output ต่ำกว่าเล็กน้อยใน validation (**10.454 เทียบกับ 10.457**). ความต่างมีขนาดเล็ก และผลทั้งสองวิธีใช้จำนวน prediction pairs เท่ากัน

### E4 — ข้อมูลจากสถานีข้างเคียง

Validation ของทุกวิธีอยู่ในช่วง **10.476–10.496** และ test อยู่ในช่วง **5.759–5.776** คะแนนจึงแทบไม่เปลี่ยนเมื่อเพิ่ม neighbor features; distance-weighted ดีขึ้นจากแบบไม่มี neighbor เพียง **0.001 RMSE** บน test ขณะที่แบบไม่ถ่วงน้ำหนักและแบบถ่วงตามลม/ระยะทางแย่ลงเล็กน้อย

## ข้อควรระวังในการเปรียบเทียบ

1. **มี dataset hash อยู่สองชุด** แม้ manifest ทั้งสองชุดระบุช่วงวัน `2024-01-15` ถึง `2026-09-18`, 192,768 แถว และ 218 สถานีเหมือนกัน:
   - `c247a167…`: E1.1–E1.3, E2.1–E2.3 และ E3.1–E3.2
   - `7e160e7b…`: E1.4–E1.5 และ E4.1–E4.4

   จำนวนแถวและช่วงวันเหมือนกันไม่ได้ยืนยันว่าเนื้อหาข้อมูลเหมือนกัน ควรตรวจสาเหตุของ hash ที่ต่างก่อนเปรียบเทียบผลข้ามสองชุดนี้
2. Validation เริ่ม `2025-11-19`; test เริ่ม `2026-04-13`. คะแนน validation กับ test แตกต่างกันมากในหลายรัน จึงควรอ่านเป็นผลคนละช่วงเวลา ไม่ควรแทนกัน
3. E3 มีจำนวน prediction pairs ต่ำกว่า E1/E2/E4 (`199,339` validation และ `215,257` test) จึงควรคำนึงถึงกลุ่มข้อมูลที่ใช้ประเมินเมื่อนำไปเทียบข้ามแกนทดลอง
4. ตารางนี้รายงาน holdout validation/test; ไม่ได้นำ walk-forward CV มาเฉลี่ยรวมกับคะแนนดังกล่าว

## การเลือกผู้ชนะ

ใช้ validation เลือกโมเดลในแต่ละแกน แล้วดู test เป็นการยืนยันผล โดยเปรียบเทียบเฉพาะรันที่ใช้ข้อมูลและกลุ่มประเมินเดียวกัน หากคะแนนต่างกันเพียงเล็กน้อยจะถือว่ายังไม่พบผู้ชนะที่ชัดเจน เพราะแต่ละการทดลองใช้ seed เดียว (`42`) และไม่มีช่วงความเชื่อมั่นสำหรับวัดว่าความต่างคงทนเพียงใด

| แกนทดลอง | ตัวเลือกนำจาก validation | สิ่งที่ test ยืนยัน | คำตัดสิน |
|---|---|---|---|
| E1 — วิธีฝึก | `E1_2` เป็นตัวเลือกนำในกลุ่ม hash `c247a167…`; `E1_4` ได้ validation ต่ำสุดโดยรวมที่ 9.316 แต่ใช้ hash `7e160e7b…` | `E1_2` ได้ 4.618; `E1_4` ได้ 4.777 บนอีก hash | เลือก `E1_2` เป็นผู้ชนะชั่วคราวในกลุ่มข้อมูลเดียวกันกับ E1.1–E1.3; ต้องรันเทียบใหม่บน hash เดียวกับ E1.4 ก่อนตัดสินรวม |
| E2 — อัลกอริทึม | `E2_2` LightGBM นำ `E2_1` เพียง 0.002 RMSE | Test ของ `E2_1`/`E2_2` ซ้ำ validation; `E2_3` มี test 4.894 แต่เทียบกับอีกสองรันไม่ได้ | ยังไม่มีผู้ชนะที่ยืนยันด้วย test; LightGBM เป็นเพียงตัวเลือกนำจาก validation |
| E3 — รูปแบบพยากรณ์ | `E3_2` Multi-output นำเพียง 0.004 RMSE | `E3_1` Direct ดีกว่าบน test เพียง 0.044 RMSE | ยังไม่พบผู้ชนะที่ชัดเจน; ความได้เปรียบสลับกันระหว่าง validation กับ test และมีขนาดเล็ก |
| E4 — Neighbor features | `E4_1` ไม่มี neighbor features ได้ validation ต่ำสุดที่ 10.476 | `E4_3` distance-weighted ดีกว่า `E4_1` เพียง 0.001 RMSE บน test | เลือก `E4_1` เป็นตัวเลือกนำจาก validation; ยังไม่มีหลักฐานว่า neighbor features เพิ่มประโยชน์จริง |

### เทียบกับ naive persistence

การเทียบกับ persistence baseline ช่วยดูว่าโมเดลเรียนรู้ได้ดีกว่าการใช้ค่าล่าสุดหรือไม่:

- `E1_2` มี test primary macro RMSE **4.618** เทียบกับ baseline **5.657** ซึ่งต่ำกว่าประมาณ **18.4%**; เป็นหลักฐานสนับสนุนการเลือก Global XGBoost ในกลุ่มข้อมูลเดียวกัน
- `E1_5` ได้ **7.131** เทียบกับ baseline **5.657** จึงแย่กว่า baseline ราว **26.1%**
- ค่า paired test ที่บันทึกไว้ใน config ของ E3 และ E4 สูงกว่า persistence ทั้งหมด: E3 ประมาณ **2.9–3.7%** และ E4 ประมาณ **1.6–1.9%** สูงกว่า จึงยังไม่ควรนำ Direct/Multi-output หรือ neighbor features ไปแทน baseline ด้วยผลชุดนี้เพียงอย่างเดียว
- ยังสรุป E2 เทียบกับ persistence บน test ไม่ได้จนกว่าจะได้ test artifacts ที่แยกจาก validation อย่างถูกต้อง

### ผู้ชนะรวมในตอนนี้

**E1.2 Global XGBoost เป็นผู้สมัครอันดับหนึ่งจากผลที่ตรวจสอบได้** เพราะชนะ E1.1 และ E1.3 บน dataset hash เดียวกันทั้ง validation และ test และมี test RMSE ต่ำสุดในผลที่ยืนยันได้ อย่างไรก็ตาม ยังไม่ควรประกาศเป็นผู้ชนะสุดท้ายของ Stage 1 จนกว่าจะ (1) รัน E1.2 เทียบกับ E1.4 บนข้อมูล hash เดียวกัน และ (2) รัน E2.1/E2.2 ใหม่เพื่อให้มี test ที่ตรวจสอบได้

## ข้อสรุปและขั้นถัดไป

แนวทางชั่วคราวคือเดินหน้าด้วย **E1.2 Global XGBoost** เป็น candidate หลัก ส่วน E2.2, E3.2 และ E4.1 เป็นตัวเลือกนำเฉพาะคะแนน validation ของแกนนั้น ๆ ยังไม่มีหลักฐานพอให้ยืนยันผู้ชนะสุดท้ายทุกแกน ควรสร้าง test artifacts ใหม่สำหรับ E2.1/E2.2 และรัน E1.2 กับ E1.4 บน dataset เดียวกันก่อนสรุปอันดับรวม

## แหล่งข้อมูล

ตัวเลขมาจากไฟล์ `*_metrics_overall.json` และ `*_metrics_overall_test.json` ในโฟลเดอร์ `result/experiment-stage-1_artifacts_E1` ถึง `E4`. นิยาม primary macro RMSE คือค่าเฉลี่ย RMSE ระหว่างคู่สถานี × ระยะพยากรณ์ตาม `experiment-stage-1/common/metrics.py`; คำอธิบายปัจจัยของการทดลองอ้างอิง `experiment-stage-1/README.md`.
