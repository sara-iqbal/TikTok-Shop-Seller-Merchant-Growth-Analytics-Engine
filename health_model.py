"""
health_model.py
----------------
Builds a "merchant health score": a classifier that predicts whether an
active seller will go dark (churn) in the next 2 months, based on their
behaviour in their most recent 3 months on the platform.

Pipeline:
  1. Feature engineering per seller as of a rolling observation point
  2. Train/test split (time-safe: features from early tenure, label from later)
  3. Gradient Boosting classifier vs Logistic Regression baseline
  4. Evaluation: ROC-AUC, precision/recall, confusion matrix
  5. Feature importance + a ranked "at-risk" leaderboard for the dashboard
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix


def build_features(sellers: pd.DataFrame, metrics: pd.DataFrame, observation_tenure=3, horizon=2):
    """
    For every seller who reached `observation_tenure` months, build features
    from months [0, observation_tenure], and the label = did they churn
    (become inactive and stay inactive) within the following `horizon` months.
    """
    m = metrics.merge(sellers, on="seller_id")

    window = m[m["tenure_months"] <= observation_tenure]
    feats = window.groupby("seller_id").agg(
        avg_gmv=("gmv", "mean"),
        gmv_std=("gmv", "std"),
        avg_orders=("orders", "mean"),
        return_rate=("returns", lambda x: x.sum() / max(window.loc[x.index, "orders"].sum(), 1)),
        avg_ads_spend=("ads_spend", "mean"),
        months_observed=("month", "count"),
    ).reset_index()

    trend = (
        window.sort_values(["seller_id", "tenure_months"])
        .groupby("seller_id")["gmv"]
        .apply(lambda s: np.polyfit(range(len(s)), s, 1)[0] if len(s) > 1 else 0)
        .rename("gmv_trend")
        .reset_index()
    )
    feats = feats.merge(trend, on="seller_id")

    static = sellers[["seller_id", "category", "country", "tier", "seller_quality"]]
    feats = feats.merge(static, on="seller_id")

    max_tenure_seen = m.groupby("seller_id")["tenure_months"].max()
    still_present_by_horizon = max_tenure_seen[max_tenure_seen >= observation_tenure + horizon].index
    eligible = feats[feats["seller_id"].isin(max_tenure_seen.index)].copy()
    eligible = eligible[eligible["seller_id"].map(max_tenure_seen) >= observation_tenure]

    label_window = m[(m["tenure_months"] > observation_tenure) & (m["tenure_months"] <= observation_tenure + horizon)]
    still_active = set(label_window["seller_id"].unique())
    eligible = eligible[eligible["seller_id"].isin(
        max_tenure_seen[max_tenure_seen >= observation_tenure].index
    )]
    eligible["churned"] = (~eligible["seller_id"].isin(still_active)).astype(int)
    # only keep sellers whose signup is old enough that the horizon has actually elapsed by "today"
    eligible = eligible[eligible["seller_id"].isin(
        sellers[sellers["signup_month"] <= pd.Timestamp("2026-06-01") - pd.DateOffset(months=observation_tenure + horizon)]["seller_id"]
    )]

    eligible["gmv_std"] = eligible["gmv_std"].fillna(0)
    return eligible.reset_index(drop=True)


def train_model(features: pd.DataFrame):
    num_cols = ["avg_gmv", "gmv_std", "avg_orders", "return_rate", "avg_ads_spend",
                "gmv_trend", "seller_quality", "months_observed"]
    cat_cols = ["category", "country", "tier"]

    X = features[num_cols + cat_cols]
    y = features["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    gbc = Pipeline([
        ("prep", preprocess),
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.06, random_state=42)),
    ])
    class_weight = {0: 1.0, 1: (y_train == 0).sum() / max((y_train == 1).sum(), 1)}
    sample_weight = y_train.map(class_weight).values
    gbc.fit(X_train, y_train, clf__sample_weight=sample_weight)

    logit = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    logit.fit(X_train, y_train)

    results = {}
    for name, model in [("gradient_boosting", gbc), ("logistic_regression", logit)]:
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "roc_auc": round(roc_auc_score(y_test, proba), 3),
            "precision": round(precision_score(y_test, pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, pred, zero_division=0), 3),
            "f1": round(f1_score(y_test, pred, zero_division=0), 3),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "test_churn_rate": round(y_test.mean(), 3),
            "n_test": len(y_test),
        }

    # feature importance from the winning model (gradient boosting)
    feature_names = num_cols + list(
        gbc.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols)
    )
    importances = gbc.named_steps["clf"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}) \
        .sort_values("importance", ascending=False).head(10)

    # score ALL sellers (train+test) for the at-risk leaderboard
    all_proba = gbc.predict_proba(X)[:, 1]
    leaderboard = features[["seller_id", "category", "country", "tier"]].copy()
    leaderboard["churn_risk_score"] = all_proba.round(3)
    leaderboard = leaderboard.sort_values("churn_risk_score", ascending=False)

    return results, importance_df, leaderboard


if __name__ == "__main__":
    sellers = pd.read_csv("data/sellers.csv", parse_dates=["signup_month"])
    metrics = pd.read_csv("data/monthly_metrics.csv", parse_dates=["month"])

    feats = build_features(sellers, metrics)
    print(f"Feature set: {len(feats)} sellers, churn rate = {feats['churned'].mean():.2%}")

    results, importance_df, leaderboard = train_model(feats)
    print("\nModel results:")
    for name, r in results.items():
        print(name, r)
    print("\nTop features:\n", importance_df)
    print("\nTop 10 at-risk sellers:\n", leaderboard.head(10))
