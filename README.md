# Five-Step Multi-Dimensional Learnability Validation Protocol (v2.0)

Reference implementation and full reproduction code for:

> Al-Ma'shumah, F., Dani, Y., Triyani, Y., & Suandi, D. (2026).
> *A Five-Step Multi-Dimensional Learnability Validation Protocol for
> Meningitis Clinical Risk Assessment.* Journal of Computing Theories and Applications.

## The protocol

| Step | Question | Pass rule (fixed in `src/config.py`) |
|---|---|---|
| 1 | Does the target carry measured dependence on the features, before any model exists? | max \|Pearson r\| > 0.10 **or** max mutual information > 0.05 nats, training fold. A failure is a **warning**; the analysis continues. |
| 2 | Do the fitted probabilities beat the prevalence model? | Brier score of the Platt-scaled LightGBM (LGBM-Cal) < uncertainty term |
| 3 | Does acting on the probabilities beat both default strategies? | LGBM-Cal net benefit exceeds max(treat-all, treat-none) by ≥ 0.01 at some threshold in [0.05, 0.50] |
| 4 | Does the model rely on a variable with measured association? | top mean \|SHAP\| feature satisfies the Step 1 rule |
| 5 | Do fitted effect directions agree with the clinical literature? | ≥ 70 % of the literature-referenced logistic regression coefficient signs agree |

Steps 2–5 are evaluated on the binary decision contrast of each target, with the
clinically actionable class as the positive class: **Bacterial** (diagnosis),
**High risk** (risk level), **Deceased** (mortality). A target that passes all five
steps is reported as *passing the five-step validation* and is eligible for
further, external validation; it is not declared ready for clinical use.

### Step 5 reference directions

A direction is scored only when at least two independent published sources report it.

| Feature | Bacterial / high risk (presentation) | Deceased (prognostic) |
|---|---|---|
| CSF WBC count | ↑ Spanos 1989; Nigrovic 2007; Alnomasy 2021 | ↓ van de Beek 2004; Bijlsma 2016; Tubiana 2020 |
| CSF protein | ↑ Spanos 1989; Nigrovic 2007; Alnomasy 2021 | not scored (one source) |
| CSF glucose | ↓ Spanos 1989; Alnomasy 2021 | ↓ Tubiana 2020; Chekrouni 2023 |
| Blood WBC count | ↑ Nigrovic 2007; Alnomasy 2021 | not scored (one source) |
| CRP | ↑ Gerdes 1998; Singh 2025 | ↑ Bijlsma 2016; Chekrouni 2023; Zhou 2025 |
| Age | not scored | ↑ van de Beek 2004; Bijlsma 2016; Tubiana 2020; Chekrouni 2023; Zhou 2025 |
| Pathogen present | not scored (one source) | ↑ positive culture: van de Beek 2004; Bijlsma 2016 |
| Hemoglobin, platelets, gender | not scored | not scored |

## Reproducing the results

```bash
git clone https://github.com/fathimahalmashumah/meningitis-learnability-protocol.git
cd meningitis-learnability-protocol
pip install -r requirements.txt
python src/run_all.py     # every analysis -> results/results.json (about 5 minutes)
python src/figures.py     # Figures 1-9 at print size -> figures/
python -m pytest tests    # consistency checks
```

Or run `S1_meningitis_analysis.ipynb` top to bottom. All seeds are fixed at 42.

| Manuscript item | Source |
|---|---|
| Section 3.2, Step 1 values; Fig. 2 | `results.json: step1`; `figures.py: fig2` |
| Table 1, Fig. 3 (repeated CV) | `results.json: cv`; `fig3_cv.png` |
| Table 2, Fig. 4 (class-wise, LightGBM) | `results.json: classwise`; `fig4_confusion.png` |
| Table 3, Figs. 5–6 (mortality calibration and DCA) | `results.json: brier_mortality, dca_mortality` |
| Table 4 (Step 5 sign audit) | `results.json: steps.*.S5` |
| Table 5 (verdict), Section 4.7 (ablation) | `results.json: steps, ablation` |
| Figs. 7–8 (SHAP), Fig. 9 (integer score) | `results.json: steps.*.S4, waterfall, integer_score` |
| XOR control, Section 4.9 | `results.json: xor_control` |

## Changes in v2.0

* One configuration module (`src/config.py`) and one implementation of each step
  (`src/protocol.py`); the verdict, the ablation and every figure call the same functions.
* Deceased is the positive class in every mortality analysis, including calibration,
  decision curves and the integer score (v1 scripts 03–05 used Recovered).
* One Step 5 rule: literature-referenced directions, 70 % agreement (v1 used 6/10 and
  7/10 agreement with training-fold correlation signs in different scripts).
* The ablation uses the final Step 1 rule (correlation **or** mutual information).
* LightGBM is the representative model for every single-fold analysis.
* Binary features now enter the integer score when present; the score's rounding bound
  is reported on the probability scale.

## Notes

The feature matrix is integer-typed, as every feature is recorded as an integer, so
SMOTE's synthetic values are rounded down to integers. The dataset is synthetic
(public Kaggle dataset `chantest/meningitis-classification`); no external cohort was
used, so the protocol is demonstrated rather than validated.

## Licence

MIT, see `LICENSE`. Please cite the article if you use this work; see `CITATION.cff`.
