"""
ab_test.py
----------
Analyzes the simulated seller-onboarding experiment with the statistical
rigor a DS team would expect:
  - Two-sample Welch's t-test on 90-day GMV (continuous metric)
  - Chi-square test on activation rate (binary metric)
  - 95% confidence intervals via normal approximation
  - Observed effect size (Cohen's d) and a post-hoc power estimate
  - A minimum-detectable-effect (MDE) calculation for the sample size collected

No external stats package (e.g. statsmodels) is used - the tests are
implemented directly on top of scipy.stats so every formula is explicit
and auditable, which matters for a metrics/experimentation role.
"""

import numpy as np
import pandas as pd
from scipy import stats


def welch_t_test(control, treatment):
    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
    n1, n2 = len(control), len(treatment)
    m1, m2 = control.mean(), treatment.mean()
    s1, s2 = control.std(ddof=1), treatment.std(ddof=1)

    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (m2 - m1) / pooled_sd

    se_diff = np.sqrt(s1**2 / n1 + s2**2 / n2)
    diff = m2 - m1
    ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    return {
        "control_mean": round(m1, 2),
        "treatment_mean": round(m2, 2),
        "abs_lift": round(diff, 2),
        "rel_lift_pct": round(diff / m1 * 100, 2),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 5),
        "cohens_d": round(cohens_d, 3),
        "ci_95_low": round(ci_low, 2),
        "ci_95_high": round(ci_high, 2),
        "significant_at_5pct": bool(p_val < 0.05),
    }


def proportion_chi_square(control_flags, treatment_flags):
    n1, n2 = len(control_flags), len(treatment_flags)
    x1, x2 = control_flags.sum(), treatment_flags.sum()
    p1, p2 = x1 / n1, x2 / n2

    table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    chi2, p_val, dof, _ = stats.chi2_contingency(table, correction=True)

    p_pool = (x1 + x2) / (n1 + n2)
    se_diff = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    diff = p2 - p1
    ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    return {
        "control_rate": round(p1, 4),
        "treatment_rate": round(p2, 4),
        "abs_lift_pp": round(diff * 100, 2),
        "rel_lift_pct": round(diff / p1 * 100, 2),
        "chi2_stat": round(chi2, 3),
        "p_value": round(p_val, 5),
        "ci_95_low_pp": round(ci_low * 100, 2),
        "ci_95_high_pp": round(ci_high * 100, 2),
        "significant_at_5pct": bool(p_val < 0.05),
    }


def post_hoc_power(control, treatment, alpha=0.05):
    """Approximate power for the observed effect size given the collected sample sizes."""
    n1, n2 = len(control), len(treatment)
    pooled_sd = np.sqrt(((n1 - 1) * control.std(ddof=1) ** 2 + (n2 - 1) * treatment.std(ddof=1) ** 2) / (n1 + n2 - 2))
    effect = (treatment.mean() - control.mean()) / pooled_sd
    n_harmonic = 2 / (1 / n1 + 1 / n2)
    ncp = effect * np.sqrt(n_harmonic / 2)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    return round(float(power), 3)


def minimum_detectable_effect(baseline_std, n_per_group, alpha=0.05, power=0.8):
    """Given the sample size actually collected, what relative effect size
    could we reliably have detected? Useful to caveat a null/borderline result."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    mde = (z_alpha + z_beta) * baseline_std * np.sqrt(2 / n_per_group)
    return round(float(mde), 2)


def run_full_analysis(ab: pd.DataFrame):
    control = ab[ab["group"] == "control"]
    treatment = ab[ab["group"] == "treatment"]

    gmv_result = welch_t_test(control["gmv_90d"].values, treatment["gmv_90d"].values)
    gmv_result["power_observed"] = post_hoc_power(control["gmv_90d"].values, treatment["gmv_90d"].values)
    gmv_result["mde_at_80pct_power"] = minimum_detectable_effect(
        ab["gmv_90d"].std(), n_per_group=len(control)
    )

    activation_result = proportion_chi_square(
        control["activated"].values.astype(int), treatment["activated"].values.astype(int)
    )

    segment_rows = []
    for cat, g in ab.groupby("category"):
        c, t = g[g["group"] == "control"], g[g["group"] == "treatment"]
        if len(c) >= 5 and len(t) >= 5:
            r = welch_t_test(c["gmv_90d"].values, t["gmv_90d"].values)
            segment_rows.append({"category": cat, "n_control": len(c), "n_treatment": len(t), **r})
    segment_df = pd.DataFrame(segment_rows)

    return {
        "sample_size": {"control": len(control), "treatment": len(treatment)},
        "gmv_90d_test": gmv_result,
        "activation_test": activation_result,
        "segment_breakdown": segment_df,
    }


if __name__ == "__main__":
    ab = pd.read_csv("data/ab_test.csv")
    results = run_full_analysis(ab)
    print("Sample sizes:", results["sample_size"])
    print("\nGMV (90d) test:", results["gmv_90d_test"])
    print("\nActivation test:", results["activation_test"])
    print("\nSegment breakdown:\n", results["segment_breakdown"])
