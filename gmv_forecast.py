"""
gmv_forecast.py
----------------
Forecasts total marketplace GMV (and the FMCG category specifically) for the
next 3 months using Holt's linear (double) exponential smoothing, implemented
directly in numpy so there's no dependency on statsmodels/prophet - every
step of the forecast is auditable.

Holt's method models a level (L) and a trend (T) that are both updated
exponentially each period:
    L_t = alpha * y_t + (1 - alpha) * (L_{t-1} + T_{t-1})
    T_t = beta  * (L_t - L_{t-1}) + (1 - beta) * T_{t-1}
    forecast(t + h) = L_t + h * T_t

alpha/beta are chosen by a small grid search minimizing one-step-ahead MAE
on a holdout of the last 4 observed months (a lightweight, explicit form of
walk-forward validation).
"""

import numpy as np
import pandas as pd


def holts_forecast(series: np.ndarray, alpha, beta, horizon):
    n = len(series)
    level = np.zeros(n)
    trend = np.zeros(n)
    level[0] = series[0]
    trend[0] = series[1] - series[0] if n > 1 else 0

    fitted = np.zeros(n)
    fitted[0] = level[0]

    for t in range(1, n):
        level[t] = alpha * series[t] + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        fitted[t] = level[t - 1] + trend[t - 1]

    forecast = np.array([level[-1] + h * trend[-1] for h in range(1, horizon + 1)])
    return fitted, forecast, level[-1], trend[-1]


def grid_search_params(series: np.ndarray, holdout=4):
    train = series[:-holdout]
    test = series[-holdout:]
    best = None
    for alpha in np.arange(0.1, 0.95, 0.1):
        for beta in np.arange(0.05, 0.95, 0.1):
            _, forecast, _, _ = holts_forecast(train, alpha, beta, holdout)
            mae = np.mean(np.abs(forecast - test))
            if best is None or mae < best[0]:
                best = (mae, alpha, beta)
    return best  # (mae, alpha, beta)


def forecast_series(monthly_totals: pd.Series, horizon=3):
    values = monthly_totals.values.astype(float)
    mae, alpha, beta = grid_search_params(values, holdout=min(4, len(values) // 5))
    fitted, forecast, level, trend = holts_forecast(values, alpha, beta, horizon)

    last_month = monthly_totals.index[-1]
    future_months = pd.date_range(last_month, periods=horizon + 1, freq="MS")[1:]

    return {
        "params": {"alpha": round(float(alpha), 2), "beta": round(float(beta), 2), "holdout_mae": round(float(mae), 1)},
        "history": {str(d.date()): round(float(v), 1) for d, v in monthly_totals.items()},
        "fitted": {str(d.date()): round(float(v), 1) for d, v in zip(monthly_totals.index, fitted)},
        "forecast": {str(d.date()): round(float(v), 1) for d, v in zip(future_months, forecast)},
        "trend_per_month": round(float(trend), 1),
    }


def build_all_forecasts(metrics: pd.DataFrame, sellers: pd.DataFrame, horizon=3):
    m = metrics.merge(sellers[["seller_id", "category"]], on="seller_id")
    m = m[m["month"] < pd.Timestamp("2026-06-01")]  # keep the last partial month out of the trend fit

    total_monthly = m.groupby("month")["gmv"].sum().sort_index()
    total_forecast = forecast_series(total_monthly, horizon)

    fmcg_monthly = m[m["category"] == "FMCG"].groupby("month")["gmv"].sum().sort_index()
    fmcg_forecast = forecast_series(fmcg_monthly, horizon)

    return {"total": total_forecast, "fmcg": fmcg_forecast}


if __name__ == "__main__":
    sellers = pd.read_csv("data/sellers.csv", parse_dates=["signup_month"])
    metrics = pd.read_csv("data/monthly_metrics.csv", parse_dates=["month"])
    forecasts = build_all_forecasts(metrics, sellers)
    print("Total GMV forecast:", forecasts["total"]["forecast"])
    print("Params:", forecasts["total"]["params"])
    print("\nFMCG GMV forecast:", forecasts["fmcg"]["forecast"])
    print("Params:", forecasts["fmcg"]["params"])
