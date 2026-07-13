"""
generate_data.py
-----------------
Generates a synthetic but realistic TikTok Shop marketplace dataset:
  - sellers.csv      : one row per seller with signup info, category, tier, country
  - monthly_metrics.csv : seller-month panel (GMV, orders, returns, ads spend, active flag)
  - ab_test.csv      : seller-level assignment + outcome for a simulated onboarding-flow experiment

The generator is seeded for reproducibility. Distributions are chosen to mimic
real e-commerce marketplace patterns: heavy-tailed GMV, category-driven return
rates (FMCG returns lower than Fashion, for example), and a churn hazard that
rises after an initial honeymoon period.
"""

import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

RNG = np.random.default_rng(42)

CATEGORIES = ["FMCG", "Beauty", "Fashion", "Home & Living", "Electronics", "Toys & Hobbies", "Sports"]
CATEGORY_WEIGHTS = [0.22, 0.20, 0.18, 0.14, 0.12, 0.08, 0.06]  # FMCG + Beauty lean common on TikTok Shop
COUNTRIES = ["UK", "DE", "FR", "ES", "IT"]
COUNTRY_WEIGHTS = [0.38, 0.22, 0.18, 0.12, 0.10]
TIERS = ["Standard", "Plus", "Enterprise"]
TIER_WEIGHTS = [0.72, 0.23, 0.05]

# Category-level base economics: (avg order value, return rate, base monthly growth)
CATEGORY_PROFILE = {
    "FMCG":           dict(aov=14,  return_rate=0.03, growth=0.045, volatility=0.10),
    "Beauty":         dict(aov=22,  return_rate=0.06, growth=0.05,  volatility=0.14),
    "Fashion":        dict(aov=28,  return_rate=0.18, growth=0.04,  volatility=0.20),
    "Home & Living":  dict(aov=35,  return_rate=0.08, growth=0.03,  volatility=0.15),
    "Electronics":    dict(aov=65,  return_rate=0.11, growth=0.025, volatility=0.18),
    "Toys & Hobbies": dict(aov=24,  return_rate=0.05, growth=0.03,  volatility=0.16),
    "Sports":         dict(aov=30,  return_rate=0.07, growth=0.035, volatility=0.15),
}

START_MONTH = date(2024, 1, 1)
END_MONTH = date(2026, 6, 1)  # 30 months of history, "today" = Jul 2026
N_SELLERS = 1400


def month_range(start, end):
    months = []
    cur = start
    while cur <= end:
        months.append(cur)
        cur = cur + relativedelta(months=1)
    return months


def generate_sellers(n=N_SELLERS):
    months = month_range(START_MONTH, END_MONTH)
    signup_months = RNG.choice(months, size=n, p=_ramp_weights(len(months)))
    categories = RNG.choice(CATEGORIES, size=n, p=CATEGORY_WEIGHTS)
    countries = RNG.choice(COUNTRIES, size=n, p=COUNTRY_WEIGHTS)
    tiers = RNG.choice(TIERS, size=n, p=TIER_WEIGHTS)

    sellers = pd.DataFrame({
        "seller_id": [f"S{100000+i}" for i in range(n)],
        "signup_month": pd.to_datetime(signup_months),
        "category": categories,
        "country": countries,
        "tier": tiers,
    })
    # baseline "quality" latent factor per seller drives both GMV level and churn risk
    sellers["seller_quality"] = RNG.beta(2.2, 2.5, size=n)  # 0..1
    return sellers


def _ramp_weights(n_months):
    """More sellers sign up in recent months (platform growth) - ramp-up weighting."""
    w = np.linspace(0.4, 1.6, n_months)
    return w / w.sum()


def generate_monthly_metrics(sellers: pd.DataFrame):
    months = month_range(START_MONTH, END_MONTH)
    month_index = {m: i for i, m in enumerate(months)}
    rows = []

    for _, s in sellers.iterrows():
        profile = CATEGORY_PROFILE[s["category"]]
        signup_i = month_index[s["signup_month"].date()]
        tier_mult = {"Standard": 1.0, "Plus": 1.8, "Enterprise": 3.4}[s["tier"]]
        base_orders = (8 + 40 * s["seller_quality"]) * tier_mult

        active = True
        # churn hazard grows with tenure unless quality is high; small early "honeymoon" grace
        for m_idx in range(signup_i, len(months)):
            tenure = m_idx - signup_i
            if not active:
                break

            category_risk = {"Fashion": 1.25, "Beauty": 1.1, "Electronics": 1.05, "Home & Living": 0.95,
                              "Toys & Hobbies": 0.95, "Sports": 0.95, "FMCG": 0.75}[s["category"]]
            hazard = 0.022 + 0.10 * (1 - s["seller_quality"]) * (1 if tenure > 2 else 0.2)
            hazard *= 1.15 if s["tier"] == "Standard" else (0.55 if s["tier"] == "Enterprise" else 0.85)
            hazard *= category_risk
            if tenure > 0 and RNG.random() < hazard:
                active = False  # churns this month (last active month recorded, then goes quiet)

            growth_noise = RNG.normal(profile["growth"], profile["volatility"] * 0.3)
            lifecycle_mult = 1 + growth_noise * min(tenure, 10) - 0.01 * max(tenure - 14, 0)
            lifecycle_mult = max(lifecycle_mult, 0.15)

            orders = max(0, RNG.negative_binomial(6, 6 / (6 + base_orders * lifecycle_mult)))
            aov = max(3, RNG.normal(profile["aov"], profile["aov"] * 0.25))
            gmv = round(orders * aov, 2)
            returns = RNG.binomial(orders, profile["return_rate"]) if orders > 0 else 0
            ads_spend = round(gmv * RNG.uniform(0.02, 0.09) * (1.3 if s["tier"] != "Standard" else 1.0), 2)

            rows.append({
                "seller_id": s["seller_id"],
                "month": pd.Timestamp(months[m_idx]),
                "tenure_months": tenure,
                "orders": int(orders),
                "gmv": gmv,
                "returns": int(returns),
                "ads_spend": ads_spend,
                "active": True,
            })

            if not active:
                break

    df = pd.DataFrame(rows)
    return df


def generate_ab_test(sellers: pd.DataFrame, metrics: pd.DataFrame):
    """
    Simulates an experiment: sellers who signed up in the last 6 months of history
    were randomized into a new streamlined onboarding flow (treatment) vs the
    legacy flow (control). Outcome: GMV in the seller's first 90 days and whether
    they became "activated" (>=1 order within 14 days).
    """
    cutoff = pd.Timestamp(END_MONTH) - pd.DateOffset(months=6)
    cohort = sellers[sellers["signup_month"] >= cutoff].copy()
    cohort["group"] = RNG.choice(["control", "treatment"], size=len(cohort), p=[0.5, 0.5])

    # treatment lifts activation and early GMV modestly, on top of natural seller_quality variance
    lift_activation = 0.06     # +6pp absolute on activation-within-14-days
    lift_gmv = 0.09            # +9% relative on 90-day GMV
    base_activation_rate = 0.74  # legacy flow: ~74% of new sellers place an order within 14 days

    first90 = (
        metrics[metrics["tenure_months"] <= 2]
        .groupby("seller_id")["gmv"].sum()
        .rename("gmv_90d")
    )

    cohort = cohort.set_index("seller_id").join(first90)
    cohort["gmv_90d"] = cohort["gmv_90d"].fillna(0)

    is_t = cohort["group"] == "treatment"

    # activation probability driven by seller_quality (latent) + flow lift for treatment
    activation_prob = base_activation_rate + 0.20 * (cohort["seller_quality"] - 0.5)
    activation_prob = activation_prob + is_t * lift_activation
    activation_prob = activation_prob.clip(0.05, 0.97)
    cohort["activated"] = RNG.random(len(cohort)) < activation_prob.values

    boost = RNG.normal(lift_gmv, 0.03, size=len(cohort))
    cohort.loc[is_t, "gmv_90d"] = cohort.loc[is_t, "gmv_90d"] * (1 + boost[is_t.values])
    # sellers who never activated have ~0 GMV regardless of group
    cohort.loc[~cohort["activated"], "gmv_90d"] = cohort.loc[~cohort["activated"], "gmv_90d"] * RNG.uniform(0, 0.15, size=(~cohort["activated"]).sum())

    cohort = cohort.reset_index()[["seller_id", "category", "country", "tier", "group", "gmv_90d", "activated"]]
    return cohort


if __name__ == "__main__":
    sellers = generate_sellers()
    metrics = generate_monthly_metrics(sellers)
    ab = generate_ab_test(sellers, metrics)

    sellers.to_csv("data/sellers.csv", index=False)
    metrics.to_csv("data/monthly_metrics.csv", index=False)
    ab.to_csv("data/ab_test.csv", index=False)

    print(f"sellers: {len(sellers)} rows")
    print(f"monthly_metrics: {len(metrics)} rows")
    print(f"ab_test: {len(ab)} rows")
