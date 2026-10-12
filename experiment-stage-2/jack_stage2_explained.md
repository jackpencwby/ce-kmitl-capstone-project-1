# Stage 2 (LightGBM) — คำอธิบายโค้ดและการแก้ไขของ Jack

## GPU support for experiments 15/16 (2026-10-12)

15/16 already import `jack_run_stage2`. They therefore use the hierarchical
grid below, and 16 uses one horizon-stacked LightGBM model for global and
one per eligible station. Both levels remain LightGBM OOF residual correction.

The runner now accepts `--device gpu` (OpenCL on Windows) and `--device cpu`.
The selected backend is passed to every global, OOF and local fit, including
test refits. Run IDs contain `LGBM_GPU` or `LGBM_CPU`; artifacts record the
requested and actual device. Both scalar and stacked probes run before data
loading. A GPU failure stops with a diagnostic instead of falling back.

CPU remains the default. On this machine with the locally built wheel, OpenCL
platform **1**, device **0** is the NVIDIA RTX 4060 Laptop GPU; automatic
selection chooses the AMD integrated GPU instead. Platform ordering is specific
to the LightGBM OpenCL loader: verify the GPU name printed by preflight rather
than copying indices from another OpenCL utility or another machine.

```powershell
# Preflight first
.venv\Scripts\python.exe experiment-stage-2\E2_15_global_local_tree_single_head.py --device gpu --gpu-platform-id 1 --gpu-device-id 0
.venv\Scripts\python.exe experiment-stage-2\E2_16_global_local_tree_multi_head.py --device gpu --gpu-platform-id 1 --gpu-device-id 0

# Full runs, one at a time
.venv\Scripts\python.exe experiment-stage-2\E2_15_global_local_tree_single_head.py --device gpu --gpu-platform-id 1 --gpu-device-id 0 --train
.venv\Scripts\python.exe experiment-stage-2\E2_16_global_local_tree_multi_head.py --device gpu --gpu-platform-id 1 --gpu-device-id 0 --train
```

GPU requires a GPU-enabled LightGBM build. The original local 4.7.0 binary
reported `GPU Tree Learner was not enabled in this build`. Windows uses
`device_type="gpu"` (OpenCL), not the Linux CUDA backend. See the
[official installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html#build-gpu-version).
GPU changes computation, not the model architecture, grids, target alignment,
OOF boundaries or the fixed-budget policy of 15/16. Floating-point results
can differ from CPU; no speedup or bit-for-bit equality is assumed. Old 16
artifacts from `MultiOutputRegressor` belong to a different architecture.

The sections below describe Jack's original CPU implementation. The historical
factory name `make_cpu_lgbm` is retained, but it now honors `device_type` passed
by the runner. Direct Python calls and experiments 1–4 retain CPU defaults.

### Local GPU installation

The project `.venv` now has LightGBM **4.7.0** rebuilt with integrated OpenCL.
The source came from the PyPI 4.7.0 sdist. Build settings were
`__INTEGRATE_OPENCL=ON`, `CMAKE_POLICY_VERSION_MINIMUM=3.5`, Visual Studio 2022
MSVC 14.38 (x64). Boost 1.74's build setup required explicit `msvc` bootstrap,
`toolset=msvc-14.3`, `address-model=64`, `architecture=x86`, the existing
`vcvars64.bat` with setup rewriting disabled, and `BOOST_ALL_NO_LIB` to use
CMake's explicit Boost libraries. These are build-system adjustments only.

A reinstallable local wheel is retained in the ignored artifacts directory:
`artifacts/builds/lightgbm-4.7.0-py3-none-win_amd64.whl`.
SHA256: `7fb77a97e32216f1d2cf88a5ecc92ed0549e1f05d2de93d3560f2056d8ff164e`.

```powershell
.venv\Scripts\python.exe -m pip install --no-deps --force-reinstall experiment-stage-2\artifacts\builds\lightgbm-4.7.0-py3-none-win_amd64.whl
```

Reinstalling the stock PyPI wheel can remove GPU support. No other Python
dependencies were replaced for this GPU installation.

เอกสารนี้อธิบายไฟล์ `jack_*` ใน `experiment-stage-2/` ว่าแต่ละไฟล์ทำอะไร, ทำงานอย่างไร และอะไรถูกเปลี่ยนจากเวอร์ชันเดิม (random search + `MultiOutputRegressor`)

## 1. ภาพรวมไฟล์

| ไฟล์ (ชื่อเดิม) | หน้าที่ |
|---|---|
| `jack_run_stage2.py` (`run_stage2.py`) | เอนจินหลักทั้งหมด: อ่าน args, สร้าง spec, เทรนโมเดล, ค้นหา hyperparameter, refit, ทดสอบ, บันทึก artifacts |
| `jack_E2_1_global_single_head.py` | Wrapper บางๆ: ใส่ `--experiment 1` แล้วเรียก `main()` |
| `jack_E2_2_global_multi_head.py` | เหมือนกัน แต่ `--experiment 2` |
| `jack_E2_3_regional_single_head.py` | `--experiment 3` |
| `jack_E2_4_regional_multi_head.py` | `--experiment 4` |
| `E2_15_*`, `E2_16_*` (ไม่ได้เปลี่ยนชื่อ) | ใช้เอนจินตัวเดียวกัน (`--experiment 15/16`); แก้เฉพาะบรรทัด import ให้ชี้ไป `jack_run_stage2` |
| `tests/jack_test_grid_and_multi_head.py` | unit test ใหม่ของ grid search และ multi-head |

ไฟล์ E2_x ทั้งสี่ไม่มี logic ของตัวเอง ทั้งหมดอยู่ใน `jack_run_stage2.py`

## 2. ตารางการทดลอง

| ID | Scope | Forecast head | จำนวนโมเดลที่เทรน (ต่อ 1 candidate) |
|---|---|---|---|
| 1 | Global (ทุกสถานีรวมกัน) | Single — แยกตาม horizon | 7 |
| 2 | Global | Multi — **โมเดลเดียว** | 1 |
| 3 | Regional (แยกตามภูมิภาค) | Single | 7 × จำนวน region |
| 4 | Regional | Multi — **โมเดลเดียวต่อ region** | 1 × จำนวน region |
| 15 | Global + residual corrector รายสถานี | Single | (เดิม ไม่เปลี่ยนโครงสร้าง) |
| 16 | Global + residual corrector รายสถานี | Multi | 1 global + 1 ต่อสถานี |

## 3. โครงสร้าง `jack_run_stage2.py`

ลำดับการทำงานของ `main()`:

1. `parse_args()` อ่าน argument
2. `require_lightgbm()` + `probe_multi_head()` ตรวจว่า LightGBM บน CPU ใช้ได้ และโมเดลเดียวให้ผลลัพธ์ shape `(N, 7)` (ถ้าไม่มี `--train` จะหยุดที่นี่ = dry run)
3. `runner.prepare_dataframe()` (โค้ด Stage 1) โหลดข้อมูล สร้าง feature/target (lag, rolling ใช้ข้อมูลอดีตเท่านั้น) — Stage 2 ไม่แตะส่วนนี้
4. `hierarchical_search()` ค้นหา hyperparameter โดยใช้ **validation เท่านั้น**
5. รันผู้ชนะบน validation อีกครั้ง → บันทึก artifacts ของ validation และเก็บ "จำนวนต้นไม้ที่ early stopping เลือก" ต่อโมเดล
6. Refit บน **train + validation** ด้วยจำนวนต้นไม้นั้น แล้วประเมินบน **test เพียงครั้งเดียว** (ข้ามได้ด้วย `--skip-test`)

ฟังก์ชันสำคัญ:

- `make_cpu_lgbm(params)` สร้าง `LGBMRegressor` แบบ `device_type="cpu"`, `subsample_freq=1`, seed จาก `--seed`
- `make_spec()` สร้าง `RunSpec` (โครงสร้างของ Stage 1) ให้แต่ละ experiment — `complete_target_evaluation=True` เฉพาะ multi (ต้องมีเป้าหมายครบ 7 ตัวต่อแถว)
- `_direct_partition()` — single-output: วนทีละ horizon เทรน 1 โมเดล/horizon (เลือกแถวแบบเดียวกับ `_fit_partition` ของ Stage 1 เพิ่มเพียง early stopping)
- `_multi_output_partition()` — multi-output: โมเดลเดียว (ดูหัวข้อ 4)
- `_run_multi_residual()` — exp 16 (global + residual รายสถานี) เรียก `_multi_output_partition` ทั้งสองระดับ
- `run_pooled()` — ตัวกลางที่แยก global / regional / residual แล้วเรียก partition ที่เหมาะสม; ตอน `evaluation="validation"` จะตัดแถวช่วง test ออกก่อนเสมอ (test locked)
- `hierarchical_search()` / `_run_phase()` / `_run_round()` — grid search (หัวข้อ 5)
- `primary_score()` — ใช้ `primary_macro_rmse` จาก `metrics.build_reports` ของ Stage 1 (ยิ่งต่ำยิ่งดี)

## 4. Multi-head: จาก 7 โมเดลใน wrapper → 1 โมเดล

**เดิม:** `MultiOutputRegressor(LGBMRegressor)` — ภายในคือ LightGBM 7 ตัวที่แยกกันเทรน แค่ห่อด้วย object เดียว

**ใหม่:** LightGBM ไม่มี multi-output tree แบบ vector-leaf (ต่างจาก XGBoost `multi_strategy="multi_output_tree"` ที่ Tawan ใช้) จึงใช้วิธี *horizon-stacked single model*:

```text
แถวเดิม  (n แถว, F feature, 7 target)
   ↓ stack
n×7 แถว: [features..., forecast_horizon = 1..7]  →  y = target_t{horizon}
   ↓
LGBMRegressor ตัวเดียว .fit() ครั้งเดียว   (ต้นไม้ชุดเดียวใช้ร่วมกันทุก horizon)
   ↓ predict บนแถวที่ stack แบบเดียวกัน
reshape (7, n).T  →  prediction shape (n, 7)
```

- ลำดับ stack เป็น horizon-major (ทุกแถวของ t+1 ก่อน แล้ว t+2 …) จึง reshape กลับได้ตรงตำแหน่ง (มี unit test ตรวจว่าคอลัมน์ j = t+(j+1))
- ต้นไม้เลือก split บน `forecast_horizon` ได้ ทำให้ต่าง horizon มี leaf ต่างกันเมื่อจำเป็น
- ไม่ใช้ `MultiOutputRegressor` และไม่มีลูป 7 โมเดลซ่อนอยู่ (ตรวจใน artifacts: exp 2 = 1 ไฟล์โมเดล, exp 4 = 1 ไฟล์/region, exp 1 = 7, exp 3 = 7/region)
- `forecast_horizon` อยู่ในรายการ feature ของโมเดล multi และใน `feature_importance.csv`
- ข้อแลกเปลี่ยน: ข้อมูลเทรนใหญ่ขึ้น 7 เท่า (RAM/เวลาต่อโมเดลสูงขึ้น) แต่เทรนแค่ 1 ครั้ง, และ loss เป็น MSE รวมทุก horizon (early stopping ก็ดูค่ารวมนี้)

## 5. Grid search (แทน random search)

โดเมนที่ให้มา (`HYPERPARAMETER_GRID`) มี 5·4·5·5·6·5·6·6·6·5 = **16,200,000** ชุด — รัน grid เต็มไม่ได้จริง จึงใช้ **Coarse-to-Fine Hierarchical Grid Search** (แบบเดียวกับที่ Tawan ใช้กับ XGBoost): ยังเป็น grid แบบกำหนดแน่นอน (deterministic) ไม่มีการสุ่ม โดยเดินทีละกลุ่มพารามิเตอร์ และตัดตัวที่แย่ทิ้งระหว่างทางแบบ successive halving

| Phase | พารามิเตอร์ที่ค้นหา | Grid | จำนวน candidate | รอบ / budget (จำนวนต้นไม้สูงสุด) → เก็บผู้รอด |
|---|---|---|---|---|
| 1 โครงสร้างหยาบ | `learning_rate`[0.03,0.1] · `num_leaves`[15,63,255] · `max_depth`[-1,7] · `min_child_samples`[10,50] · `subsample`[0.7,1.0] · `colsample_bytree`[0.6,0.8,1.0] (regularization ตรึง: `min_split_gain=0, reg_alpha=0, reg_lambda=1`) | 2·3·2·2·2·3 | **144** | 300 → เก็บ ⌈1/3⌉ (48) → 800 → เก็บ ⌈1/3⌉ (16) → 2000 |
| 2 Regularization | `min_split_gain`(5) × `reg_alpha`(6) × `reg_lambda`(6) ครบทุกค่าในโดเมน; โครงสร้างตรึงที่ผู้ชนะ Phase 1 | 5·6·6 | **180** | 800 → เก็บ 36 → 2000 |
| 3 Fine local | ค่าข้างเคียง 3 ค่าในโดเมนเดิมของ 6 พารามิเตอร์โครงสร้างรอบผู้ชนะ Phase 2 (ขอบโดเมนจะเลื่อนหน้าต่าง) | 3⁶ | **729** | 300 → เก็บ 72 → 800 → เก็บ 12 → 2000 |

- **`n_estimators`**: ไม่ใช่แกน grid อิสระ — ค่า 300/800/2000 (ซึ่งอยู่ในโดเมน) ใช้เป็น *งบจำนวนต้นไม้* ของแต่ละรอบ และ **early stopping (100 รอบ, ดู validation) เป็นตัวเลือกจำนวนต้นไม้จริง** (บันทึกเป็น `best_iteration`) ใช้กับ exp 1–4; exp 15/16 ไม่ใช้ early stopping (มี OOF ซ้อน) ใช้ `n_estimators = budget` ตรงๆ
- ผู้ชนะสุดท้าย = candidate ที่ได้ `primary_macro_rmse` ต่ำสุดในบรรดาผลที่รันที่ budget เต็ม (2000) ของทุก phase (ผลเสมอแก้ด้วยการเรียงสตริงพารามิเตอร์ → deterministic)
- Regional (exp 3/4): ใช้ชุดพารามิเตอร์ **ร่วมกันทุก region** ที่เลือกจาก macro RMSE รวม (ไม่จูนแยก region) เพื่อให้เทียบกับ Global ได้เป็นธรรม
- Single vs Multi เลือกพารามิเตอร์ของตัวเองด้วยโปรโตคอลเดียวกัน (ไม่บังคับให้เหมือนกัน)
- จำนวนการประเมินในโหมดเต็ม: Phase 1 = 144+48+16 = 208, Phase 2 = 180+36 = 216, Phase 3 = 729+72+12 = 813 → **1,237 candidate evaluations**; จำนวน fit = 1,237 × (7 / 1 / 7×region / 1×region ตาม exp 1/2/3/4)
- ไม่มีการเข้าถึง test ระหว่างจูน: `run_pooled(..., "validation")` ตัดแถว `date >= TEST_START` ออกทิ้ง

### Arguments ใหม่

```text
--seed (เดิม --random-seed ยังใช้ได้)   --early-stopping-rounds 100
--phase1-pruning / --no-phase1-pruning
--max-phase1-candidates N  --max-phase2-candidates N  --max-phase3-candidates N   (ใช้ตอนพัฒนา/ทดสอบ)
--development   โหมด smoke test: candidate ≤3 ต่อ phase, budget เล็ก (20–60 ต้นไม้), ข้าม test, artifacts ขึ้นต้น DEV_
--n-trials   ถูกลบออก (ไม่มี random search แล้ว)
```

## 6. Artifacts (`artifacts/<run_id>__<timestamp>/`)

`tuning_trials.csv` (ทุกการประเมิน: phase, round, trial_id, parameters, boosting_budget, best_iteration, validation_primary_macro_rmse, rmse_t1..t7, elapsed_seconds, status, failure_reason) · `phase1/2/3_results.csv` · `tuning_summary.json` · `best_params.json` (`selected_phase`, `primary_macro_rmse`, `selected_boosting_rounds`, `params`) · `config.json` · `dataset_manifest.json` · metrics / predictions ของ validation และ test · `feature_importance*.csv` · `model/<split>/` (โมเดล .joblib + manifest)

> `best_params.json` เปลี่ยนรูปแบบ: เดิมเป็น dict ของพารามิเตอร์ตรงๆ ตอนนี้พารามิเตอร์อยู่ใต้คีย์ `params`

## 7. วิธีรัน

```powershell
# dry run (ตรวจ LightGBM + multi-head probe)
.venv\Scripts\python.exe experiment-stage-2\jack_E2_2_global_multi_head.py

# smoke test บนข้อมูลจริง (บาง station)
.venv\Scripts\python.exe experiment-stage-2\jack_E2_2_global_multi_head.py --train --source gcs --max-stations 10 --development

# รันเต็ม (ทีละ process)
.venv\Scripts\python.exe experiment-stage-2\jack_E2_1_global_single_head.py --train --source gcs
```

Unit tests:

```powershell
.venv\Scripts\python.exe -m unittest experiment-stage-2/tests/jack_test_grid_and_multi_head.py experiment-stage-2/tests/test_residual.py
```

## 8. ผลการทดสอบที่ทำแล้ว

**ข้อมูลสังเคราะห์** (ก่อนมี `.env`): `--development` exp 1, 2, 15, 16 และโปรโตคอลเต็มแบบจำกัด candidate exp 1–4 ผ่าน

**ข้อมูลจริงจาก GCS** (ดาวน์โหลดแล้วแคชเป็น CSV ในเครื่อง, 227 ไฟล์สถานี, 218 สถานีที่ eligible):

- `--development --max-stations 10` ผ่านทั้ง exp 1, 2, 3, 4, 15, 16 (ใช้เวลา ~30–190 วินาทีต่อ exp)
- โปรโตคอลเต็มแบบจำกัด candidate (`--max-phase*-candidates 2`, budget จริง 300/800/2000, early stopping 50) บน 12 สถานี ครบ 4 ภูมิภาค: exp 1, 2, 4 ผ่านถึงขั้น refit train+val และประเมิน test; จำนวนไฟล์โมเดลที่ test = 7 / 1 / 4 (หนึ่งโมเดลต่อ region)

| exp | val primary macro RMSE | test primary macro RMSE | naive (persistence) test |
|---|---|---|---|
| 1 Global single | 8.93 | 4.67 | 5.34 |
| 2 Global multi (1 โมเดล) | 9.12 | 4.70 | 5.28 |
| 4 Regional multi (1 โมเดล/region) | 8.73 | 6.10 | 5.28 |

ตัวเลขนี้มาจากการจูนแค่ 2 candidate ต่อ phase บน 12 สถานี **จึงไม่ใช่ผลการทดลองจริง** ใช้ยืนยันว่าท่อทำงานถูกต้องเท่านั้น (naive คำนวณบนกลุ่มแถวที่จับคู่กับโมเดลแต่ละ exp จึงต่างกันเล็กน้อย) exp 4 แย่กว่า naive ในรอบนี้เพราะบาง region มีสถานีแค่ 3 แห่งและ early stopping หยุดที่ 14–116 ต้นไม้ — ต้องดูอีกครั้งในรันเต็ม

- unit tests 15 ข้อผ่าน (รวม `test_residual.py` ที่แก้จำนวน local fit ของ exp 16 จาก 14 เป็น 2 เพราะตอนนี้ 1 โมเดล/สถานี)

## 9. ข้อควรระวัง

- **ยังไม่เคยรันโปรโตคอลเต็มกับทุกสถานี** — ควรวัดเวลา/RAM ก่อนโดยเฉพาะ exp 2/4/16 ที่ข้อมูลเทรนขยาย 7 เท่า
- ผลลัพธ์ exp 2/4/16 ไม่เทียบตรงกับรอบก่อนหน้า (เปลี่ยนสถาปัตยกรรมโมเดล) และผล exp 1–4 ก็ไม่เทียบตรงกับ random search เดิม (เปลี่ยนวิธีจูน + มี early stopping)
- Phase 3 (729 candidate) บน CPU เป็นส่วนที่แพงที่สุด ถ้าต้องการลดเวลาให้ปรับ `PHASE_KEEP` / `PHASE_BUDGETS` ที่หัวไฟล์
- `requirements.txt` ระบุ `lightgbm>=4.0` โค้ดทดสอบกับ LightGBM 4.7.0, pandas 3.0.6, scikit-learn 1.9.1
- ไฟล์ `README.md`, `Tawan_*` ที่ไม่เกี่ยวข้องไม่ได้ถูกแตะ (README แก้เฉพาะส่วนที่ล้าสมัย)
