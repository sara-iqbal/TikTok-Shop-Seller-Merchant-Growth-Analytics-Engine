"""
build_dashboard_data.py
------------------------
Runs the full analytics pipeline end to end and serializes everything the
dashboard needs into output/dashboard_data.json. This is the single
integration point between the Python analysis layer and the static HTML
dashboard (the HTML embeds this JSON directly, so it needs zero server).
"""

import json
import numpy as np
import pandas as pd

from cohort_analysis import build_cohort_retention, category_retention_summary
from ab_test import run_full_analysis
from health_model import build_features, train_model
from gmv_forecast import build_all_forecasts


def to_native(obj):
    """Recursively convert numpy/pandas scalar types to plain Python for json.dumps."""
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def main():
    sellers = pd.read_csv("data/sellers.csv", parse_dates=["signup_month"])
    metrics = pd.read_csv("data/monthly_metrics.csv", parse_dates=["month"])
    ab = pd.read_csv("data/ab_test.csv")

    print("Building cohort retention...")
    cohort_pivot, cohort_size = build_cohort_retention(sellers, metrics, max_tenure=9)
    cat_retention = category_retention_summary(sellers, metrics)

    print("Running A/B test analysis...")
    ab_results = run_full_analysis(ab)

    print("Training merchant health model...")
    features = build_features(sellers, metrics)
    model_results, importance_df, leaderboard = train_model(features)

    print("Forecasting GMV...")
    forecasts = build_all_forecasts(metrics, sellers)

    # ---- top-line KPIs ----
    latest_full_month = metrics[metrics["month"] < pd.Timestamp("2026-06-01")]["month"].max()
    last3 = metrics[metrics["month"] >= latest_full_month - pd.DateOffset(months=2)]
    total_gmv_ttm = metrics[metrics["month"] >= latest_full_month - pd.DateOffset(months=11)]["gmv"].sum()
    active_sellers_latest = metrics[metrics["month"] == latest_full_month]["seller_id"].nunique()
    avg_return_rate = round(metrics["returns"].sum() / max(metrics["orders"].sum(), 1) * 100, 2)
    fmcg_share = round(
        metrics.merge(sellers[["seller_id", "category"]], on="seller_id")
        .pipe(lambda d: d[d["category"] == "FMCG"]["gmv"].sum() / d["gmv"].sum() * 100), 2
    )

    kpis = {
        "total_sellers": int(sellers["seller_id"].nunique()),
        "active_sellers_latest_month": int(active_sellers_latest),
        "gmv_ttm": round(float(total_gmv_ttm), 0),
        "avg_return_rate_pct": avg_return_rate,
        "fmcg_gmv_share_pct": fmcg_share,
        "latest_month": str(latest_full_month.date()),
    }

    # ---- cohort heatmap payload ----
    cohort_payload = {
        "months": [str(m.date()) for m in cohort_pivot.index],
        "tenure_cols": [int(c) for c in cohort_pivot.columns],
        "matrix": [[None if pd.isna(v) else round(float(v), 3) for v in row] for row in cohort_pivot.values],
        "cohort_sizes": {str(k.date()): int(v) for k, v in cohort_size.items()},
        "category_retention_m3": cat_retention.reset_index().to_dict(orient="records"),
    }

    # ---- A/B test payload ----
    ab_payload = to_native({
        "sample_size": ab_results["sample_size"],
        "gmv_90d_test": ab_results["gmv_90d_test"],
        "activation_test": ab_results["activation_test"],
        "segment_breakdown": ab_results["segment_breakdown"].to_dict(orient="records"),
    })

    # ---- health model payload ----
    health_payload = to_native({
        "model_results": model_results,
        "feature_importance": importance_df.to_dict(orient="records"),
        "at_risk_leaderboard": leaderboard.head(15).to_dict(orient="records"),
        "churn_rate_observed": round(float(features["churned"].mean()), 4),
        "n_sellers_scored": int(len(features)),
    })

    payload = {
        "generated_at": "2026-07-13",
        "kpis": to_native(kpis),
        "cohort": to_native(cohort_payload),
        "ab_test": ab_payload,
        "health_model": health_payload,
        "forecast": to_native(forecasts),
    }

    with open("output/dashboard_data.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Wrote output/dashboard_data.json")
    print(json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()
