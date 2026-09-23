# Five-Step Multi-Dimensional Learnability Validation Protocol

Reference implementation and full reproduction code for:

> Al-Ma'shumah, F., Dani, Y., Triyani, Y., & Suandi, D. (2026).
> *A Five-Step Multi-Dimensional Learnability Validation Protocol for
> Meningitis Clinical Risk Assessment.*
> Journal of Computing Theories and Applications.

## What this is

Clinical machine learning is usually judged on discrimination alone. A model
that separates classes well is reported as successful, even when its
probabilities are miscalibrated, its explanations point at variables with no
measured association with the outcome, and nobody has checked whether the
target can be learned from the available features at all.

This repository implements a protocol that tests a prediction target on five
independent axes:

| Step | Question it answers |
|---|---|
| 1 | Does the target carry measured dependence on the features, before any model exists? |
| 2 | Do the fitted probabilities correspond to observed frequencies? |
| 3 | Does acting on those probabilities do more good than harm? |
| 4 | Are the variables the model relies on the ones a clinician would expect? |
| 5 | Do the fitted effect directions agree with the clinical audit reference? |

A target is *learnable* when Step 1 passes. It is *deployment-ready* only when
all five pass.

## Data

`data/meningitis.csv` is a copy of the public Kaggle dataset
[`chantest/meningitis-classification`](https://www.kaggle.com/datasets/chantest/meningitis-classification):
1,200 records, 14 clinical variables, three prediction targets (aetiological
diagnosis, mortality outcome, clinical risk level). The data are synthetic in
origin. Nothing here is patient data.

## Reproducing the results

```bash
git clone https://github.com/<ACCOUNT>/meningitis-learnability-protocol.git
cd meningitis-learnability-protocol
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/02_leakage_check_and_cv.py        # Step 1 on the training fold; repeated CV
python src/03_classwise_and_ablation.py      # class-wise metrics; protocol ablation
python src/01_five_step_protocol.py          # the five-step verdict per target
python src/05_correlation_and_confusion.py   # correlation matrix; confusion matrices
python src/04_generate_figures.py            # Figures 4, 5 and 9
```

Or open `S1_meningitis_analysis.ipynb` and run it top to bottom.

All random seeds are fixed at 42. Outputs land in `results/` and `figures/`.

## Which script produces which result

| Manuscript item | Script |
|---|---|
| Table 1, discrimination under repeated CV | `src/02_leakage_check_and_cv.py` |
| Table 2, class-wise metrics | `src/03_classwise_and_ablation.py` |
| Table 3, Brier decomposition | `src/05_correlation_and_confusion.py` |
| Table 4, five-step verdict | `src/01_five_step_protocol.py` |
| Section 4.6, ablation | `src/03_classwise_and_ablation.py` |
| Fig. 2, correlation matrix (training fold) | `src/05_correlation_and_confusion.py` |
| Fig. 3, confusion matrices | `src/05_correlation_and_confusion.py` |
| Figs. 4, 5, 9 | `src/04_generate_figures.py` |
| XOR control, Section 4.8 | `src/01_five_step_protocol.py` |

Figures 1, 6, 7 and 8 are the workflow diagram and the SHAP plots; the SHAP
plots come from the notebook.

## Version pinning matters

`requirements.txt` pins **XGBoost to the 2.0 series**. Tree-construction
defaults changed in 3.x, and under XGBoost 3.x the mortality-target MCC shifts
from +0.013 to −0.017 and the AUC from 0.548 to 0.542. Two scikit-learn APIs
used by the original analysis were removed in 1.7 and 1.8
(`LogisticRegression(multi_class=...)` and `CalibratedClassifierCV(cv='prefit')`);
the code guards both by version, so it runs on current scikit-learn without
changing behaviour.

## Applying the protocol to your own data

The protocol is model-agnostic and nothing in it is specific to meningitis.
To reuse it, replace `data/meningitis.csv`, then edit the feature list and the
target definitions at the top of `src/01_five_step_protocol.py`. Steps 2, 3
and 5 need a binary decision, so multiclass targets are reduced to the binary
contrast that carries the clinical decision; the reduction is set in the
`BIN` dictionary in the same file.

Step 1 thresholds are `tau = 0.10` for absolute correlation and `0.05` nats
for mutual information. Both are working conventions, not derived quantities.

## Limitations

The dataset is synthetic and no external cohort was used, so the protocol is
demonstrated rather than validated. The Step 1 screen uses pairwise statistics
and will miss dependence that lives in higher-order interactions; the
exclusive-or control in `src/01_five_step_protocol.py` shows a case where a
correlation-only screen would have come close to rejecting a target that
LightGBM recovers perfectly.

## Licence

MIT, see `LICENSE`. Please cite the paper if you use this work; see
`CITATION.cff`.
