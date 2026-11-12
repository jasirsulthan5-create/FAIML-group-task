"""Module 3 - Simple ML process flow: telecom customer-churn prediction.
Follows the flowchart: collect -> clean -> features -> split -> choose
algorithm -> train + k-fold CV -> test once -> evaluate.
Data is SYNTHETIC (replace load_data() with a real dataset)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = np.random.default_rng(42)


def load_data(n=5000):
    """1. Data collection (synthetic billing / usage / support data)."""
    df = pd.DataFrame({
        "avg_bill_6m": RNG.normal(60, 20, n).clip(10),
        "data_usage_change": RNG.normal(0, 1, n),
        "complaints_90d": RNG.poisson(0.6, n),
        "tenure_months": RNG.integers(1, 72, n),
        "contract_months_left": RNG.integers(0, 24, n),
        "contract_type": RNG.choice(["monthly", "1-year", "2-year"], n,
                                    p=[0.5, 0.3, 0.2]),
    })
    logit = (-2.45 + 0.03 * (df.avg_bill_6m - 60) - 0.5 * df.data_usage_change
             + 0.6 * df.complaints_90d - 0.03 * df.tenure_months
             + 1.0 * (df.contract_type == "monthly"))
    p = 1 / (1 + np.exp(-logit))
    df["churn"] = (RNG.random(n) < p).astype(int)
    df.loc[RNG.choice(n, 100, replace=False), "avg_bill_6m"] = np.nan  # missing
    return df


def build_preprocessor():
    """2-3. Clean (impute), encode categoricals, scale numerics."""
    num = ["avg_bill_6m", "data_usage_change", "complaints_90d",
           "tenure_months", "contract_months_left"]
    cat = ["contract_type"]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def main():
    df = load_data()
    print(f"Churn rate: {df.churn.mean():.1%}")
    X, y = df.drop(columns="churn"), df["churn"]

    # 4. Stratified 80:20 split (test set untouched until the end)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # 5-6. Candidate algorithms + 5-fold cross-validation on training data
    candidates = {
        "Logistic regression": LogisticRegression(max_iter=1000,
                                                  class_weight="balanced"),
        "Random forest": RandomForestClassifier(n_estimators=200,
                                                class_weight="balanced",
                                                random_state=42),
        "Gradient boosting": GradientBoostingClassifier(random_state=42),
    }
    best_name, best_score = None, -1
    for name, model in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", model)])
        score = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc").mean()
        print(f"{name:20s} CV ROC-AUC = {score:.3f}")
        if score > best_score:
            best_name, best_score = name, score

    # 7. Train best model, test ONCE on held-out data
    final = Pipeline([("prep", build_preprocessor()),
                      ("clf", candidates[best_name])]).fit(X_tr, y_tr)
    pred = final.predict(X_te)
    proba = final.predict_proba(X_te)[:, 1]

    # 8. Evaluation
    print(f"\nSelected model: {best_name}")
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, target_names=["stays", "churns"]))
    print(f"Test ROC-AUC: {roc_auc_score(y_te, proba):.3f}")


if __name__ == "__main__":
    main()
