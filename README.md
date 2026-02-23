# Lending Club Loan Default Prediction

Predicting whether a borrower will repay their loan using only information available at the time of the loan decision.

## Dataset

- **Source:** LendingClub loan data (~887K loans, 74 features)
- **Target:** `loan_performance` — 1 (Fully Paid) vs 0 (Charged Off)
- **Note:** The CSV file (`loans (1).csv`) is not included due to size (463 MB). Place it in the project folder before running.

## Approach

### Feature Selection
Following domain knowledge, we only use features available at loan origination. All post-loan features (payment history, recoveries, current balances, etc.) are excluded to avoid data leakage.

### Feature Engineering
- Ordinal encoding for `sub_grade` (A1=1 to G5=35) preserving risk order
- Credit history length from `earliest_cr_line`
- Target encoding for categorical variables (addr_state, purpose, etc.)
- Derived features: installment-to-income ratio, DTI x interest rate, total interest cost, delinquency flags

### Models
All models trained with GridSearchCV (10-fold CV, AUC scoring):
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting (HistGradientBoostingClassifier)
- XGBoost

### Results
Best test AUC: ~0.80 (Ensemble / XGBoost)

## How to Run

1. Place `loans (1).csv` in the project folder
2. Open `HW1_Final.ipynb` in Jupyter Notebook
3. Select **Python 3 (ipykernel)** kernel
4. Run all cells (Kernel → Restart & Run All)

## Requirements

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost (optional — requires `brew install libomp` on Mac)
