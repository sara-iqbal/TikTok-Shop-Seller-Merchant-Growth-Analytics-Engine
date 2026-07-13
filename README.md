# Merchant Growth Engine — TikTok Shop Seller Analytics

A self-serve analytics project simulating the kind of work a Data Science /
Business Analytics team supports for TikTok Shop's EMEA seller ecosystem:
retention diagnostics, an onboarding-flow A/B test, a churn-risk model, and a
short-term GMV forecast — all surfaced in a live, static dashboard.

**[Live dashboard →]([dashboard/index.html](https://sara-iqbal.github.io/TikTok-Shop-Seller-Merchant-Growth-Analytics-Engine/))** (open directly in a browser, no server needed)

Built as a portfolio project for TikTok Shop Data Science / Business
Analytics / FMCG intern & graduate roles (London, 2026). It's designed to
mirror the actual problems those job descriptions call out: merchant
lifecycle and cohort analysis, GMV expansion diagnostics, experimentation,
and "adopting AI in daily workstreams" for scalable analysis.

---

## Why this project, and why this shape

I built this after reading four TikTok Shop / TikTok LIVE EMEA job postings.
Three of them (DS Graduate, Business Analytics Intern, FMCG Data Analyst
Intern) point at the same underlying skill set even though the job titles
differ:

| Sr.no | What it asks for | Where this project covers it |
|---|---|---|
| 1 | A/B testing, statistical modeling, metrics development, big-data mining | `src/ab_test.py` — Welch's t-test, chi-square, power analysis, MDE |
| 2 | Merchant lifecycle, cohort analysis, GMV expansion, governance health | `src/cohort_analysis.py` — retention cohorts, category-level checkpoints |
| 3 | Category performance tracking, ad-hoc analysis, operational efficiency | FMCG is modeled as its own category throughout; dashboard breaks it out explicitly |
| 4 | Segmentation, predictive "high-potential" scoring, OKR dashboards | `src/health_model.py` uses the same segmentation/scoring pattern, applied to sellers instead of creators |

Rather than building four disconnected toy demos, I built one coherent
dataset and pipeline that a merchant-ops or DS team could plausibly run,
and let each analysis answer a real question a stakeholder would ask.

## What's in here

```
tiktok-merchant-analytics/
├── main.py                      # single entrypoint: reproduces everything
├── requirements.txt
├── src/
│   ├── generate_data.py         # synthetic seller/order data generator
│   ├── cohort_analysis.py       # retention cohorts
│   ├── ab_test.py               # onboarding-flow experiment analysis
│   ├── health_model.py          # churn-risk classifier
│   ├── gmv_forecast.py          # Holt's exponential smoothing forecast
│   └── build_dashboard_data.py  # orchestrates all of the above -> JSON
├── data/                        # generated CSVs (sellers, monthly metrics, A/B test)
├── output/dashboard_data.json   # single JSON payload consumed by the dashboard
└── dashboard/
    ├── template.html            # dashboard shell with a data placeholder
    └── index.html               # final, self-contained dashboard (data inlined)
```

Run everything from scratch:

```bash
pip install -r requirements.txt
python main.py
# then open dashboard/index.html in a browser
```

## The data

There is no TikTok-internal data here TikTok Shop doesn't publish
seller-level data, so `generate_data.py` builds a **synthetic** marketplace:
1,400 sellers across 7 categories (weighted toward FMCG and Beauty, which
the job postings themselves flag as focus areas), 5 EU/UK markets, and 3
account tiers, over 30 months. Order volume, AOV, return rates, and churn
hazard are parameterized per category so the data behaves like a real
marketplace (e.g., FMCG has lower AOV and lower returns than Fashion) rather
than being uniform noise. Everything is seeded (`numpy.random.default_rng(42)`)
for reproducibility.

## Methodology

**Cohort retention.** Sellers are grouped by signup month; for each cohort I
track the share still placing orders at month 0, 1, 2… 9. This is the
standard lifecycle view merchant-ops teams use to judge onboarding quality
and spot when churn typically sets in.

**A/B test.** Sellers who joined in the last 6 months of the dataset were
randomized into a legacy vs. streamlined onboarding flow. I used:
- Welch's t-test (unequal variance) on 90-day GMV, with a 95% CI and Cohen's d
- A chi-square test on 14-day activation rate
- A post-hoc power calculation and a minimum-detectable-effect (MDE)
  calculation, so a non-significant result is reported honestly rather than
  hidden the actual result here (below) needed this caveat.

No `statsmodels`/`prophet` everything is implemented directly on `scipy.stats`
and plain `numpy` so every formula is auditable line by line.

**Churn-risk model.** A Gradient Boosting classifier (with a Logistic
Regression baseline) predicts whether an active seller goes dark within 2
months, using only their first 3 months of behavior: GMV level and trend,
order volume, return rate, ad spend, and seller tier/category. Churn is a
rare event (~6.8% base rate) I used class-balanced sample weights rather
than oversampling, and evaluated with ROC-AUC / precision / recall rather
than accuracy, since accuracy is meaningless on an imbalanced label.

**GMV forecast.** A 3-month-ahead forecast using Holt's linear (double)
exponential smoothing, implemented by hand in `numpy`. Smoothing parameters
(α, β) are chosen by grid search against a 4-month holdout a lightweight
form of walk-forward validation instead of trusting default parameters.

## Results (this run)

- **Activation lift is real and significant:** the streamlined onboarding
  flow lifted 14-day activation by **+12.1pp** (70.9% → 83.0%, p = 0.002).
- **GMV lift is directionally positive but not yet significant:** +13.2%
  (p = 0.21). The observed power at this sample size is only ~0.24, and the
  minimum detectable effect at 80% power would need a larger lift than what
  was observed the honest read is "this experiment needs a larger sample
  or a longer runtime before making a GMV-based launch decision," even
  though activation alone is a good enough reason to ship.
- **Churn model:** ROC-AUC 0.61 (Gradient Boosting) vs. 0.51 (Logistic
  Regression baseline). Top predictive features are return rate, ad spend,
  and order volume trend not raw GMV level. An AUC in the 0.6 range on a
  genuinely noisy, low-base-rate label is realistic; I'd rather report that
  honestly than tune a synthetic dataset until it produces an unrealistic
  0.9+ AUC that wouldn't reflect what a first pass at this problem actually
  looks like in production.
- **FMCG retains best:** ~97.6% of FMCG sellers are still active at the
  month-3 checkpoint, the highest of any category consistent with lower
  return rates and more repeatable purchase behavior, and a direct,
  actionable data point for prioritizing FMCG brand incubation.
- **Forecast:** total marketplace GMV projected to grow from ~€1.01M to
  ~€1.14M over the next 3 months (~6% month-over-month), driven by seller
  base growth outpacing the modest churn hazard.

## What I'd do differently with real data / more time

- Replace the synthetic churn label with a proper survival-analysis
  formulation (time-to-churn, censoring-aware) instead of a fixed 2-month
  binary window.
- Add a sequential-testing view for the A/B test so the experiment can be
  monitored (and potentially stopped early) rather than analyzed once at a
  fixed horizon.
- Swap Holt's smoothing for a model that can use seller-count and
  seasonality as explicit regressors, since GMV growth here is really being
  driven by an expanding seller base, not organic per-seller growth.
- Wire the health-model leaderboard to actually trigger a lifecycle
  intervention (e.g., an ops queue) rather than sit in a static dashboard.

## What I learned building this

Building the whole pipeline not just a model in a notebook forced a lot
of decisions a single "fit a classifier" exercise skips: how to define a
churn label without leaking the future into the features, why accuracy is
the wrong metric under class imbalance, and why an A/B test with a
directionally positive but non-significant result is actually a *more*
useful thing to present honestly than a p-hacked significant one. Writing
the dashboard data export as a single JSON contract between the Python
layer and a dependency-free static HTML file was also a good constraint —
it meant the "dashboard" could be pushed to GitHub Pages with zero backend,
which is closer to how a lot of internal analytics tooling actually ships.

## Stack

Python (pandas, numpy, scipy, scikit-learn) for the pipeline · vanilla
HTML/CSS/JS + Chart.js for the dashboard · no paid APIs, no external data,
fully reproducible from `python main.py`.
