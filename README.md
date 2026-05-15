# Lending Club — Loan-Default Prediction with No Data Leakage

> **A loan-default prediction model trained on 887K LendingClub loans — but only using features that would have been available *at the time of the loan decision*. Best test AUC 0.80, with a deliberate eye on the production realities of credit underwriting.**

**887K loans · 74 features · 5 model families benchmarked · No data leakage · ML coursework, MSBA**

---

## The Problem

The standard student approach to LendingClub data is to throw every column at XGBoost, hit AUC 0.95, and call it a day. **That model is useless in production** — most of the high-AUC features (payment history, recoveries, current balance) only exist *after* the loan is funded. They leak the answer.

The actually interesting problem is harder:

> **"At the moment a loan application arrives, can we predict whether it will be repaid — using only what we know at that moment?"** This is the question a real credit-underwriting team has to answer.

---

## Users & Jobs-to-be-Done

| User | Job-to-be-Done | Today's Workaround | Pain |
|------|----------------|--------------------|------|
| **Underwriting analyst** | When a new application arrives, I want a calibrated default-risk score I can trust. | Rules + FICO bucket | Misses signal in long-tail features |
| **Risk manager** | When I review a portfolio, I want to know which segments are over- vs. under-priced. | Lagging KPI dashboards | Reactive, not predictive |
| **Product manager (lending)** | When I design a new product, I want feature importances I can build the application UX around. | Gut + competitor mimicry | No structured signal |

---

## The Solution

A clean, leakage-free modeling pipeline that:

1. **Drops all post-funding features** — payment history, recoveries, balances, etc.
2. **Engineers credit-domain features** that an underwriter would actually compute (installment-to-income ratio, credit-history length, DTI × interest-rate interaction, total interest cost, delinquency flags)
3. **Encodes carefully** — ordinal for `sub_grade` (A1=1 → G5=35) to preserve risk order; target encoding for high-cardinality categoricals (state, purpose)
4. **Benchmarks 5 model families** with `GridSearchCV` (10-fold CV, AUC scoring): Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, XGBoost

### Key product decisions (and the tradeoffs)

| Decision | What I picked | What I rejected | Why |
|----------|---------------|-----------------|-----|
| **No data leakage** (drop post-funding columns) | A "boring" 0.80 AUC | A "fake" 0.95 AUC | A leaky model would get fired on the first portfolio cohort. Honesty about the modelable signal is the whole point. |
| **Ordinal sub_grade encoding** | A1=1 → G5=35 | One-hot or label encoding | Preserves the meaningful risk ordering; trees can split on it cleanly; LR gets a monotone signal. |
| **Multiple model families benchmarked** | All 5 evaluated under same CV | "Just use XGBoost" | Logistic regression performance gives you a *baseline*. If LR gets 0.78 and XGBoost 0.80, the marginal gain from gradient boosting is real but small — and a real underwriting team might choose LR for explainability. |
| **AUC + 10-fold CV scoring** | AUC, not accuracy | Accuracy | Defaults are imbalanced (~15% of loans). Accuracy is misleading; AUC measures rank-ordering of risk, which is what underwriters actually need. |

---

## Impact & Metrics

| Metric | Result |
|--------|--------|
| Best test AUC | **~0.80** (Ensemble / XGBoost) |
| Logistic Regression AUC | ~0.78 (baseline — close to XGBoost; explainability win) |
| Loans modeled | 887,000 |
| Features used | Origination-time only (~25 after engineering, from ~74 raw) |
| Cross-validation | 10-fold, AUC scoring, GridSearchCV |

---

## What I'd Build Next

| Priority | Feature | Why this, why now |
|----------|---------|-------------------|
| **P0** | **Calibration analysis (reliability curve)** | AUC measures rank-ordering, but underwriting needs *calibrated probabilities* (a 10% predicted default should default 10% of the time). Calibration plots + Platt scaling or isotonic regression turn AUC into a usable price. |
| **P0** | **Profitability-aware model selection** | The right model isn't the highest-AUC — it's the one that maximizes *expected portfolio return* given the cost of false positives (rejected good borrowers) and false negatives (defaults). Re-rank models by expected $ rather than AUC. |
| **P1** | **Fairness audit by protected attributes** | A real underwriting model must be tested for disparate impact. The dataset has limited protected attributes, but the *methodology* — and the conversation about where to stop modeling and start regulating — is critical. |

**What I would NOT build next:** Squeeze AUC from 0.80 to 0.81 with a fancier model. The diminishing returns are real; calibration and fairness are where the next real lift lives.

---

## My Role

**Group homework** for the ML course (MSBA, SCU).

**What I personally owned:**
- The "no leakage" principle — pushed the team to drop the easy features and accept a lower (but defensible) AUC
- Feature engineering (DTI × interest, installment-to-income, credit history length)
- Sub-grade ordinal encoding decision
- Notebook structure and writeup

---

## What I Learned

- **High AUC is suspicious.** When you see a 0.95 AUC on a credit dataset, the first question is "what leaked?" Most public ML notebooks on LendingClub don't pass this sniff test.
- **Baselines matter.** A logistic regression at 0.78 makes XGBoost at 0.80 less impressive — and might be the right *production* choice for explainability. Always run the simple model first.
- **Encoding choices encode domain knowledge.** Treating sub_grade as ordinal (vs. one-hot) bakes in the obvious-but-true fact that A grades are safer than G grades. The right encoding is the right hypothesis.
- **AUC ≠ deployable.** This was the biggest takeaway. AUC tells you the model can rank-order risk; calibration tells you whether the probabilities are usable as a price. A PM working with ML teams needs to ask the calibration question.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.x |
| Data | pandas, NumPy |
| Modeling | scikit-learn (Logistic, Tree, RF, HistGradientBoosting), XGBoost |
| Validation | GridSearchCV, 10-fold CV, AUC scoring |
| Visualization | matplotlib, seaborn |

---

## How to Run

1. Place `loans (1).csv` (~463 MB, not in repo) in the project folder
2. Open `HW1_Final.ipynb` in Jupyter
3. Select **Python 3 (ipykernel)** kernel
4. Run all cells (Kernel → Restart & Run All)

> XGBoost on Mac requires `brew install libomp`.

---

## Files

- `HW1_Final.ipynb` — Complete analysis (EDA, feature engineering, model comparison, results)
- `LoanDataDictionary.xlsx` — Reference for the 74 raw columns
- `Group Homework 1 (2).pdf` — Original assignment brief

---

**Built by [Srinidhi Jagannathan](https://github.com/sjagannathan17)** · Santa Clara University MSBA · [LinkedIn](https://linkedin.com/in/srinidhi-jagannathan) · srinidhi.jagan11@gmail.com
