import argparse
import os

import numpy as np
import pandas as pd

from .low_risk_factor import load_sp500_tickers, load_prices_close, load_spy_series


def summarize_array(vals, name):
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [f"{name}: no finite values"]

    q = np.nanpercentile(vals, [0.1, 1, 5, 50, 95, 99, 99.9])
    lines = [
        f"{name}: count={vals.size}",
        f"{name}: min={np.nanmin(vals):.6f} max={np.nanmax(vals):.6f}",
        (
            f"{name}: p0.1={q[0]:.6f} p1={q[1]:.6f} p5={q[2]:.6f} "
            f"p50={q[3]:.6f} p95={q[4]:.6f} p99={q[5]:.6f} p99.9={q[6]:.6f}"
        ),
    ]
    return lines


def price_pair(prices, dt, ticker):
    try:
        pos = prices.index.get_loc(dt)
    except KeyError:
        return None
    if pos == 0:
        return None
    prev_dt = prices.index[pos - 1]
    prev_price = prices.at[prev_dt, ticker]
    curr_price = prices.at[dt, ticker]
    return prev_dt, prev_price, curr_price


def main():
    p = argparse.ArgumentParser(description="Diagnose low risk factor data issues")
    p.add_argument("--h5", default="sp500.h5")
    p.add_argument("--spy-cache", default="data/spy.csv")
    p.add_argument("--factor", default="output/low_risk_factor_returns.csv")
    p.add_argument("--extreme", type=float, default=1.0, help="abs return threshold")
    p.add_argument("--max-events", type=int, default=20)
    p.add_argument("--out", default="output/diagnostics.txt")
    args = p.parse_args()

    report = []

    tickers = load_sp500_tickers(args.h5)
    report.append(f"tickers_in_list: {len(tickers)}")

    prices = load_prices_close(args.h5, tickers)
    report.append(f"prices_shape: {prices.shape}")
    report.append(
        f"prices_date_range: {prices.index.min().date()} -> {prices.index.max().date()}"
    )

    missing = prices.isna().mean().sort_values(ascending=False)
    report.append("missing_ratio_top10:")
    for t, v in missing.head(10).items():
        report.append(f"  {t}: {v:.2%}")

    non_positive = (prices <= 0).sum().sort_values(ascending=False)
    report.append("non_positive_price_top10:")
    for t, v in non_positive.head(10).items():
        report.append(f"  {t}: {int(v)}")

    returns = prices.pct_change()
    flat = returns.to_numpy().ravel()
    report.extend(summarize_array(flat, "stock_daily_returns"))

    extreme = returns.stack()
    extreme = extreme[np.abs(extreme) > args.extreme]
    report.append(f"extreme_events_count(abs>{args.extreme}): {len(extreme)}")
    if not extreme.empty:
        extreme = extreme.reindex(extreme.abs().sort_values(ascending=False).index)
        report.append("top_extreme_events:")
        for (dt, t), r in extreme.head(args.max_events).items():
            pair = price_pair(prices, dt, t)
            if pair is None:
                report.append(f"  {dt.date()} {t} return={r:.6f}")
            else:
                prev_dt, prev_price, curr_price = pair
                report.append(
                    f"  {dt.date()} {t} return={r:.6f} prev={prev_price} curr={curr_price}"
                )

    # SPY sanity
    start = prices.index.min().date().isoformat()
    end = (prices.index.max() + pd.Timedelta(days=1)).date().isoformat()
    try:
        spy = load_spy_series(start, end, args.spy_cache)
        spy_ret = spy.pct_change().dropna()
        report.append(
            f"spy_date_range: {spy.index.min().date()} -> {spy.index.max().date()}"
        )
        report.extend(summarize_array(spy_ret.to_numpy(), "spy_daily_returns"))
    except Exception as e:
        report.append(f"spy_load_error: {e}")

    # Factor returns sanity if present
    if os.path.exists(args.factor):
        factor = pd.read_csv(args.factor, index_col=0, parse_dates=True)
        if factor.shape[1] >= 1:
            series = factor.iloc[:, 0].dropna()
            report.append(
                f"factor_date_range: {series.index.min().date()} -> {series.index.max().date()}"
            )
            report.extend(summarize_array(series.to_numpy(), "factor_daily_returns"))
            large = series[np.abs(series) > args.extreme]
            report.append(
                f"factor_extreme_count(abs>{args.extreme}): {len(large)}"
            )
            if not large.empty:
                report.append("factor_top_extremes:")
                for dt, r in large.reindex(large.abs().sort_values(ascending=False).index).head(
                    args.max_events
                ).items():
                    report.append(f"  {dt.date()} return={r:.6f}")
    else:
        report.append("factor_file_missing")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
