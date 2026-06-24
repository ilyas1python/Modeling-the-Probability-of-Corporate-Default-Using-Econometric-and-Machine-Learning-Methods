# Modeling the Probability of Corporate Default

### Econometric vs. Machine Learning Methods on the Polish Companies Bankruptcy Dataset

A bachelor's research paper (ICEF, HSE University) comparing classical accounting-based
bankruptcy models against modern machine-learning classifiers under a single, unified,
leakage-free evaluation protocol.

---

## Overview

This project asks a question that the bankruptcy-prediction literature usually answers
too quickly: **does machine learning really beat classical econometric models, and if so,
why?** Most existing comparisons give the ML model more features, handle class imbalance
differently across models, or report only one metric. This paper removes those confounds.
It re-estimates the Altman (1968), Ohlson (1980), and Zmijewski (1984) models and compares
them with LASSO, Ridge, Random Forest, and XGBoost using **the same stratified folds, the
same fold-internal imputation, the same class weighting, and the same metrics** for every
model.

The dataset is the **Polish Companies Bankruptcy** dataset (Zięba, Tomczak & Tomczak, 2016;
[UCI ML Repository ID 365](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)):
~43,000 firm-year observations, 64 financial ratios, five forecasting horizons (1 to 5 years
ahead).

## Three contributions

1. **Missingness is an informative distress signal.** Missing financial-ratio values are not
   just a nuisance to impute away. Whether a firm reports a ratio is itself one of the
   strongest predictors of bankruptcy in the data — the three largest LASSO coefficients are
   missingness flags, and they rank high in the XGBoost SHAP importances.

2. **The ML advantage decomposes into two channels.** Using the same feature set for every
   model, the gain splits into *feature breadth* (access to 75 ratios + missingness flags vs.
   the textbook 5–7) and *nonlinearity* (thresholds and interactions). Feature breadth does
   most of the work; nonlinearity adds a further premium only once the broad feature set is
   available.

3. **A unified, leakage-free benchmark.** Every preprocessing step is fit inside the training
   fold only. The comparison reports discrimination (ROC-AUC, PR-AUC, KS), calibration (Brier,
   ECE, reliability diagrams), and economic cost — not just a single accuracy number.

---

## Headline results (1-year-ahead horizon)

| Model | Type | ROC-AUC | PR-AUC |
|---|---|:---:|:---:|
| Altman (5 ratios) | Classical | 0.772 | 0.265 |
| Ohlson (7+OENEG) | Classical | 0.795 | 0.297 |
| Zmijewski (3) | Classical | 0.776 | 0.286 |
| LASSO logit (75) | Regularized linear | 0.895 | 0.514 |
| Ridge logit (75) | Regularized linear | 0.877 | 0.448 |
| Random Forest | Tree ensemble | 0.938 | 0.627 |
| **XGBoost** | **Tree ensemble** | **0.960** | **0.810** |

The progression is clean: classical models cluster at **0.77–0.80**, regularized linear models
on the full feature set jump to **0.88–0.90**, and the tree ensembles reach **0.94–0.96**. At a
common operating threshold, XGBoost flags distressed firms with **83% precision and 68% recall**,
versus Ohlson's 32% and 46%.

### Benchmarked against the published literature

Our XGBoost (0.960) sits at the **top of the published range** for this dataset, marginally above
the best result of Zięba et al. (2016) (EXGB, 0.955) and above their plain XGBoost (0.951). Our
Ohlson baseline (0.795) lands in the same band as their LDA (0.796). The value added here is not a
higher AUC — the ceiling is already reached — but the **first PR-AUC, calibration, and unified-protocol
comparison** on this dataset.

---

## Key findings in detail

**Feature breadth vs. nonlinearity.** Holding the algorithm fixed, moving a logit from 5 to 75
features lifts ROC-AUC by ~12 points. Adding XGBoost's nonlinearity on top of the full feature set
adds ~7 more. Even a **PCA logit** (75 features compressed to ~30 uncorrelated components, removing
all multicollinearity) reaches only 0.890 — still well short of XGBoost. The remaining gap is
nonlinearity, not feature count or correlation structure.

**Performance fades with the horizon.** A pooled cross-horizon model with horizon dummies and
ratio×horizon interactions shows that the predictive content of the standard ratios is concentrated
close to the event. Profitability's effect more than halves from the 1-year to the 4-year horizon
(likelihood-ratio test: χ² = 115.3, df = 12, p < 10⁻¹⁸). This explains why simple accounting-ratio
models collapse at longer horizons while tree ensembles hold up.

**Calibration is separable from discrimination.** The class-weighted linear models are badly
miscalibrated out of the box but rank firms reasonably; isotonic recalibration removes almost all
calibration error without changing the ranking. On the irreducible sorting power, the tree advantage
is permanent.

---

## Figures

| | |
|---|---|
| ![ROC curves](figures/fig_roc_h1.png) | ![Precision-recall](figures/fig_pr_h1.png) |
| **ROC curves** at the 1-year horizon | **Precision-recall curves** — the gap is far larger here |
| ![Feature grid](figures/fig_featuregrid.png) | ![Horizon decay](figures/fig_horizon.png) |
| **Feature breadth vs. nonlinearity** decomposition | **Performance across prediction horizons** |
| ![SHAP](figures/fig_shap_bar.png) | ![Literature benchmark](figures/fig_litbenchmark.png) |
| **XGBoost SHAP importances** | **Our results vs. published benchmarks** |

---

## Repository structure

```
.
├── README.md                          # this file
├── paper/
│   └── bankruptcy_default.tex         # full LaTeX source of the paper
├── notebook/
│   └── bankruptcy_default.ipynb       # complete, runnable analysis notebook
├── code/
│   └── bankruptcy_pipeline.py         # linear Python export of the notebook
└── figures/                           # all figures (PDF + PNG)
```

## Reproducing the results

```bash
# install dependencies
pip install numpy pandas scikit-learn xgboost statsmodels scipy \
            matplotlib shap imbalanced-learn ucimlrepo

# run the full pipeline (downloads the dataset from UCI automatically)
python code/bankruptcy_pipeline.py
```

Or open `notebook/bankruptcy_default.ipynb` and run the cells top to bottom. The dataset is fetched
automatically from the UCI ML Repository; no manual download is needed.

## Methods at a glance

- **Models:** Altman, Ohlson, Zmijewski (re-estimated logits/probit); LASSO, Ridge, and PCA logits
  on all 75 features; Random Forest; XGBoost.
- **Evaluation:** stratified 5-fold cross-validation; median imputation, scaling, and any resampling
  fit inside the training fold only (leakage-free); isotonic recalibration on an inner holdout.
- **Metrics:** ROC-AUC, PR-AUC, Kolmogorov–Smirnov; Brier score with reliability/resolution/uncertainty
  decomposition; expected calibration error; recall/precision at the F1-optimal threshold; normalized
  economic cost under a 35:1 false-negative-to-false-positive ratio.
- **Interpretation:** SHAP values for the XGBoost model; LASSO coefficient signs; pooled cross-horizon
  logit with horizon dummies and slope interactions.

## Data and citation

The dataset is from:

> Zięba, M., Tomczak, S. K., & Tomczak, J. M. (2016). Ensemble boosted trees with synthetic features
> generation in application to bankruptcy prediction. *Expert Systems with Applications*, 58, 93–101.

If you use this work, please cite the paper (see `paper/bankruptcy_default.tex` for the full reference list).

## License

Released under the MIT License. The underlying dataset is distributed by the UCI Machine Learning
Repository under its own terms.
