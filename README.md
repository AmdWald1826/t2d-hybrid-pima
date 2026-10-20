# t2d-hybrid-pima

Evaluation code for **"A Hybrid Predictive Framework for Proactive Type 2 Diabetes
Management: Taxonomy and Evaluation of Statistical, Uncertainty-Based and Hybrid Classifiers"**
(E. Berber, A. Boualem; submitted to *Computer Methods in Biomechanics and Biomedical Engineering*).

The code regenerates every number, table and figure of the evaluation section (Section VI):
Table IV, Table V and Figs. 4-7.

## Quick start

```bash
pip install -r requirements.txt
python scripts/get_data.py         # downloads the public PIMA file and checks its SHA-256
python scripts/run_all.py          # PIMA evaluation (~1 h on one CPU core)
python scripts/synthetic_cohort.py # synthetic cohort, Table V, Figs. 4-7 (~1 min)
```

## PIMA evaluation (`scripts/run_all.py`)

| Item | Setting |
|---|---|
| Inputs | Glucose, blood pressure, BMI, age (all 8 attributes for the sensitivity check) |
| Missing values | zeros in glucose, blood pressure, skin thickness, insulin and BMI treated as missing; median imputation fitted on the training fold |
| Resampling | stratified 5-fold cross-validation repeated 10 times (50 splits, `random_state=2026`), identical splits for all models |
| Logistic regression | L2 penalty, C = 1, L-BFGS, z-score standardization fitted on the training fold |
| Expert fuzzy system | triangular low/high membership functions at clinical cut-points (`EXPERT_MF` in `src/t2d_models.py`); 6 rules; output = normalized firing strength of the diabetic rules |
| Tuned fuzzy system | zero-order TSK, 2 Gaussian MFs per input (16 rules), 32 parameters fitted by differential evolution on the cross-entropy (population 12 x 32, 80 generations, seed 2026 + split index), training fold only |
| Hybrid | mean of the LR and tuned-fuzzy probabilities |
| Statistics | Nadeau-Bengio corrected 95% CI and corrected resampled t-test; TOST equivalence (corrected variance) |
| Threshold | 0.5 on the predicted probability |

Expert rule base. Diabetic: (G high AND BMI high) OR (G high AND Age high) OR (G high AND BP high AND BMI high).
Non-diabetic: (G low) OR (BMI low AND Age low) OR (BP low AND BMI low).

Results (Table IV, PIMA rows; %, mean ± SD over 50 splits [corrected 95% CI]; counts are means per repetition):

| Model | Accuracy | Precision | Recall | F1 | AUC | TN/FP/FN/TP |
|---|---|---|---|---|---|---|
| Majority | 65.1 ± 0.2 [64.9, 65.3] | 0.0 ± 0.0 [0.0, 0.0] | 0.0 ± 0.0 [0.0, 0.0] | 0.0 ± 0.0 [0.0, 0.0] | 50.0 ± 0.0 [50.0, 50.0] | 500/0/268/0 |
| LR | 76.6 ± 3.1 [73.4, 79.8] | 71.0 ± 6.0 [64.8, 77.3] | 56.2 ± 7.1 [48.8, 63.7] | 62.5 ± 5.6 [56.7, 68.3] | 82.7 ± 3.3 [79.2, 86.1] | 438/62/117/151 |
| Fuzzy-expert | 72.8 ± 3.5 [69.1, 76.4] | 59.4 ± 4.4 [54.8, 64.1] | 69.8 ± 6.7 [62.8, 76.7] | 64.1 ± 4.7 [59.1, 69.0] | 77.7 ± 3.7 [73.9, 81.6] | 372/128/81/187 |
| Fuzzy-tuned | 75.9 ± 3.5 [72.2, 79.6] | 69.0 ± 7.2 [61.4, 76.5] | 57.3 ± 7.2 [49.7, 64.8] | 62.3 ± 5.8 [56.2, 68.3] | 83.2 ± 3.4 [79.7, 86.7] | 429/71/114/154 |
| Hybrid | 76.5 ± 2.7 [73.6, 79.4] | 71.1 ± 5.6 [65.3, 76.9] | 55.4 ± 6.8 [48.4, 62.5] | 62.1 ± 5.2 [56.6, 67.5] | 83.9 ± 3.1 [80.6, 87.2] | 439/61/119/149 |

Other checks (`results/summary.json`):

* LR, all 8 attributes: accuracy 76.7%, AUC 83.7% (corrected p = 0.87 vs 4 attributes).
* LR without glucose (blood pressure, BMI, age): accuracy 68.3%, AUC 73.3%.
* Tuned fuzzy minus LR, accuracy: -0.7 points, 90% CI [-3.1, 1.7]; TOST p = 0.002 (margin ±5) and 0.06 (margin ±3).
* Hybrid vs LR, AUC: corrected p = 0.058; tuned fuzzy vs LR, accuracy: p = 0.62.

## Synthetic cohort and decision rules (`scripts/synthetic_cohort.py`)

* Cohort: N = 1,000, `numpy.random.default_rng(2026)`, independent uniform inputs
  (glucose U(60, 220) mg/dL, diastolic BP U(50, 110) mmHg, BMI U(15, 45) kg/m2,
  age U(20, 80) years); label = ADA rule (glucose >= 126 mg/dL) with 5% of labels flipped
  at random. Saved as `results/synthetic_cohort.csv`. Glucose is also an input, so the
  cohort only illustrates label leakage.
* Same 10 x 5 stratified cross-validation as PIMA; no noise is added to model outputs.
* Decision rules (Table V): deterministic count of four criteria (glucose >= 126 mg/dL,
  diastolic BP >= 90 mmHg, BMI >= 30, age >= 45; positive if at least two); naive Bayes on
  training-fold tertiles (Laplace smoothing, training-fold prior); Dempster-Shafer
  combination of the expert high/low memberships on the frame {T2D, non-T2D} (pignistic
  probability); logistic regression as reference.
* Illustrative cost per person (scenario assumptions): positive prediction $750,
  false negative $3,200, true negative $0. Risk level = mean predicted score.

Table IV, synthetic rows:

| Model | Accuracy | Precision | Recall | F1 | AUC | TN/FP/FN/TP |
|---|---|---|---|---|---|---|
| Majority | 60.9 ± 0.2 [60.7, 61.1] | 60.9 ± 0.2 [60.7, 61.1] | 100.0 ± 0.0 [100.0, 100.0] | 75.7 ± 0.2 [75.5, 75.9] | 50.0 ± 0.0 [50.0, 50.0] | 0/391/0/609 |
| LR | 93.7 ± 1.4 [92.3, 95.2] | 94.6 ± 1.4 [93.1, 96.1] | 95.2 ± 2.2 [93.0, 97.5] | 94.9 ± 1.2 [93.6, 96.1] | 94.0 ± 1.9 [92.0, 96.0] | 358/33/29/580 |
| Fuzzy-expert | 76.6 ± 2.8 [73.7, 79.5] | 87.4 ± 2.3 [85.0, 89.9] | 71.9 ± 3.9 [67.8, 76.0] | 78.9 ± 2.8 [75.9, 81.8] | 84.1 ± 3.0 [81.0, 87.2] | 328/63/171/438 |

Table V (mean ± SD over the 50 splits):

| Method | Risk level | Cost ($) | Accuracy |
|---|---|---|---|
| Deterministic | 0.51 ± 0.02 | 856 ± 55 | 0.713 ± 0.028 |
| Bayesian | 0.61 ± 0.02 | 557 ± 31 | 0.907 ± 0.022 |
| Dempster-Shafer | 0.58 ± 0.02 | 1023 ± 63 | 0.658 ± 0.031 |
| LR (reference) | 0.61 ± 0.02 | 553 ± 31 | 0.937 ± 0.014 |

## Files

* `src/t2d_models.py` - data loading, models, statistics
* `scripts/get_data.py` - download and checksum of the PIMA file
* `scripts/run_all.py` - PIMA evaluation; writes `results/table_IV_pima.csv`, `results/splits_*.csv`, `results/summary.json`
* `scripts/synthetic_cohort.py` - synthetic cohort, Table IV (synthetic rows), Table V, Figs. 4-7
* `results/` - all outputs reported in the paper
* `figures/` - Figs. 4-7 (EPS, PDF, PNG)

## Licence

MIT. The PIMA data are not redistributed here; `scripts/get_data.py` downloads them from a
public mirror.
