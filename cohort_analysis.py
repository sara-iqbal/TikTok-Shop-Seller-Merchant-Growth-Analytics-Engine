"""
cohort_analysis.py
-------------------
Classic cohort retention analysis: group sellers by signup month, and for
each cohort compute the % of sellers still placing orders in month 0, 1, 2...N
after signup. This is the standard "merchant lifecycle" view requested by
marketplace/governance teams to judge onboarding and retention quality.
"""

import pandas as pd


def build_cohort_retention(sellers: pd.DataFrame, metrics: pd.DataFrame, max_tenure=12):
    cohort_size = sellers.groupby("signup_month")["seller_id"].nunique().rename("cohort_size")

    active_by_tenure = (
        metrics[metrics["tenure_months"] <= max_tenure]
        .merge(sellers[["seller_id", "signup_month"]], on="seller_id")
        .groupby(["signup_month", "tenure_months"])["seller_id"]
        .nunique()
        .rename("active_sellers")
        .reset_index()
    )

    active_by_tenure = active_by_tenure.merge(cohort_size, on="signup_month")
    active_by_tenure["retention_rate"] = (
        active_by_tenure["active_sellers"] / active_by_tenure["cohort_size"]
    ).round(4)

    pivot = active_by_tenure.pivot(index="signup_month", columns="tenure_months", values="retention_rate")
    pivot = pivot.sort_index()
    return pivot, cohort_size


def category_retention_summary(sellers: pd.DataFrame, metrics: pd.DataFrame, tenure_checkpoint=3):
    """Category-level retention at a fixed tenure checkpoint (e.g. month 3) -
    useful for spotting which verticals (FMCG, Fashion, etc.) retain merchants best."""
    eligible = sellers[sellers["signup_month"] <= pd.Timestamp("2026-03-01")]
    cohort_size = eligible.groupby("category")["seller_id"].nunique()

    active = (
        metrics[metrics["tenure_months"] == tenure_checkpoint]
        .merge(eligible[["seller_id", "category"]], on="seller_id")
        .groupby("category")["seller_id"].nunique()
    )
    summary = pd.DataFrame({
        "cohort_size": cohort_size,
        "active_at_checkpoint": active,
    }).fillna(0)
    summary["retention_rate"] = (summary["active_at_checkpoint"] / summary["cohort_size"]).round(4)
    return summary.sort_values("retention_rate", ascending=False)


if __name__ == "__main__":
    sellers = pd.read_csv("data/sellers.csv", parse_dates=["signup_month"])
    metrics = pd.read_csv("data/monthly_metrics.csv", parse_dates=["month"])

    pivot, cohort_size = build_cohort_retention(sellers, metrics)
    print(pivot.round(2))
    print("\nCategory retention @ month 3:")
    print(category_retention_summary(sellers, metrics))
