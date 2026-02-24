import argparse
import os

import numpy as np
import pandas as pd

from .low_risk_factor import (
    build_scores,
    clean_prices,
    load_prices_and_volume,
    load_prices_close,
    load_sector_map,
    load_sp500_tickers,
    load_spy_series,
    month_end_rebalance_dates,
    standardize_signals,
)


def compute_beta_vol_counts(window_returns, market_window, min_obs):
    betas = {}
    total_vols = {}
    resid_vols = {}
    counts = {}

    m = market_window.values
    for ticker in window_returns.columns:
        s = window_returns[ticker].values
        mask = ~np.isnan(s) & ~np.isnan(m)
        n_obs = int(mask.sum())
        counts[ticker] = n_obs
        if n_obs < min_obs:
            betas[ticker] = np.nan
            total_vols[ticker] = np.nan
            resid_vols[ticker] = np.nan
            continue

        s_m = s[mask]
        m_m = m[mask]
        var_m = np.var(m_m, ddof=1)
        if var_m == 0 or np.isnan(var_m):
            betas[ticker] = np.nan
            total_vols[ticker] = np.nan
            resid_vols[ticker] = np.nan
            continue

        cov = np.cov(s_m, m_m, ddof=1)[0, 1]
        beta = cov / var_m
        betas[ticker] = beta
        total_vols[ticker] = np.std(s_m, ddof=1)
        resid = s_m - beta * m_m
        resid_vols[ticker] = np.std(resid, ddof=1)

    return pd.Series(betas), pd.Series(total_vols), pd.Series(resid_vols), pd.Series(counts)


def parse_args():
    p = argparse.ArgumentParser(description="Export factor inputs per rebalance date")
    p.add_argument("--h5", default="sp500.h5")
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--skip", type=int, default=21)
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument("--method", default="combined", choices=["beta", "vol", "combined"])
    p.add_argument(
        "--side",
        default="long_short",
        choices=["long_short", "long_only", "short_only"],
    )
    p.add_argument("--spy-cache", default="data/spy.csv")
    p.add_argument("--output", default="output_adj/factor_inputs.csv")
    p.add_argument("--price-floor", type=float, default=0.1)
    p.add_argument("--price-ceil", type=float, default=1e5)
    p.add_argument("--return-cap", type=float, default=1.0)
    p.add_argument(
        "--price-field",
        default=None,
        help="preferred price column (e.g., adjusted_close, Close)",
    )
    p.add_argument("--sector-map", default="ticker_to_sector.json")
    p.add_argument("--winsor-p", type=float, default=0.01)
    p.add_argument(
        "--sector-neutral",
        action="store_true",
        default=True,
        help="apply sector-neutral z-scoring (default)",
    )
    p.add_argument(
        "--no-sector-neutral",
        dest="sector_neutral",
        action="store_false",
        help="disable sector-neutral z-scoring (use global z-scores)",
    )
    p.add_argument(
        "--weighting",
        default="equal",
        choices=["equal", "value"],
        help="equal or value (avg dollar volume) weights",
    )
    p.add_argument(
        "--min-history-frac",
        type=float,
        default=0.8,
        help="minimum fraction of lookback observations required",
    )
    p.add_argument(
        "--min-dollar-vol",
        type=float,
        default=None,
        help="minimum average dollar volume over lookback window",
    )
    p.add_argument(
        "--vol-type",
        default="total",
        choices=["total", "residual"],
        help="use total or residual (idiosyncratic) volatility",
    )
    p.add_argument(
        "--bab-beta-neutral",
        action="store_true",
        default=False,
        help="apply BAB-style beta-neutral scaling on long-short weights",
    )
    p.add_argument(
        "--bab-target-gross",
        type=float,
        default=2.0,
        help="target gross exposure after BAB scaling (None to skip)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    tickers = load_sp500_tickers(args.h5)
    sector_map = load_sector_map(args.sector_map)
    need_volume = args.weighting == "value" or args.min_dollar_vol is not None
    if need_volume:
        prices, volumes = load_prices_and_volume(
            args.h5, tickers, price_field=args.price_field
        )
    else:
        prices = load_prices_close(args.h5, tickers, price_field=args.price_field)
        volumes = None
    prices = clean_prices(prices, price_floor=args.price_floor, price_ceil=args.price_ceil)

    returns = prices.pct_change().dropna(how="all")
    if args.return_cap is not None:
        returns = returns.where(returns.abs() <= args.return_cap)

    start = returns.index.min().date().isoformat()
    end = (returns.index.max() + pd.Timedelta(days=1)).date().isoformat()
    spy = load_spy_series(start, end, args.spy_cache)
    mkt_returns = spy.pct_change().reindex(returns.index)

    common = returns.index.intersection(mkt_returns.index)
    returns = returns.loc[common]
    mkt_returns = mkt_returns.loc[common]

    rebal_dates = month_end_rebalance_dates(returns.index)

    rows = []
    for i, dt in enumerate(rebal_dates):
        pos = returns.index.get_loc(dt)
        end_pos = pos - args.skip
        start_pos = end_pos - args.lookback
        if start_pos < 0:
            continue

        window = returns.iloc[start_pos:end_pos]
        market_window = mkt_returns.iloc[start_pos:end_pos]

        min_obs = max(30, int(args.min_history_frac * args.lookback))
        betas, total_vols, resid_vols, counts = compute_beta_vol_counts(
            window, market_window, min_obs
        )
        vols = resid_vols if args.vol_type == "residual" else total_vols
        betas_w, vols_w, beta_z, vol_z = standardize_signals(
            betas, vols, sector_map, args.winsor_p, sector_neutral=args.sector_neutral
        )
        score = build_scores(beta_z, vol_z, args.method)

        score = score.dropna()
        if score.empty:
            continue

        # Liquidity filter based on average dollar volume in the lookback window
        avg_dollar_vol = None
        if args.min_dollar_vol is not None or args.weighting == "value":
            vol_window = volumes.iloc[start_pos:end_pos]
            price_window = prices.iloc[start_pos:end_pos]
            avg_dollar_vol = (price_window * vol_window).mean()
            if args.min_dollar_vol is not None:
                liquid = avg_dollar_vol[avg_dollar_vol >= args.min_dollar_vol].index
                score = score.loc[score.index.intersection(liquid)]

        n = int(len(score) * args.quantile)
        if n < 1:
            continue

        longs = set(score.nsmallest(n).index)
        shorts = set(score.nlargest(n).index)

        if args.side == "long_only":
            shorts = set()
        elif args.side == "short_only":
            longs = set()

        weights = {}
        if args.weighting == "value":
            if avg_dollar_vol is None:
                raise RuntimeError("value weighting requires volume data")
            long_vals = avg_dollar_vol.reindex(longs).dropna()
            short_vals = avg_dollar_vol.reindex(shorts).dropna()
            for t in score.index:
                weights[t] = 0.0
            if args.side == "long_short":
                if long_vals.sum() > 0:
                    for t, v in long_vals.items():
                        weights[t] = v / long_vals.sum()
                if short_vals.sum() > 0:
                    for t, v in short_vals.items():
                        weights[t] = -v / short_vals.sum()
            elif args.side == "long_only":
                if long_vals.sum() > 0:
                    for t, v in long_vals.items():
                        weights[t] = v / long_vals.sum()
            elif args.side == "short_only":
                if short_vals.sum() > 0:
                    for t, v in short_vals.items():
                        weights[t] = -v / short_vals.sum()
        else:
            for t in score.index:
                if t in longs:
                    weights[t] = 1.0 / n
                elif t in shorts:
                    weights[t] = -1.0 / n
                else:
                    weights[t] = 0.0

        # optional BAB beta-neutral scaling (long-short only)
        if args.bab_beta_neutral and args.side == "long_short":
            w_ser = pd.Series(weights)
            beta_slice = betas.reindex(w_ser.index)
            long_mask = w_ser > 0
            short_mask = w_ser < 0
            if long_mask.any() and short_mask.any():
                beta_long = (w_ser[long_mask] * beta_slice[long_mask]).sum()
                beta_short = (w_ser[short_mask] * beta_slice[short_mask]).sum()
                if beta_short != 0:
                    scale_short = -beta_long / beta_short
                    w_ser.loc[short_mask] = w_ser.loc[short_mask] * scale_short
                if args.bab_target_gross is not None:
                    gross = w_ser.abs().sum()
                    if gross > 0:
                        w_ser = w_ser * (args.bab_target_gross / gross)
            weights = w_ser.to_dict()

        rank = score.rank(ascending=True, method="average")

        lookback_start = returns.index[start_pos]
        lookback_end = returns.index[end_pos - 1]
        hold_start = returns.index[pos + 1] if pos + 1 < len(returns.index) else returns.index[pos]
        hold_end = (
            rebal_dates[i + 1]
            if i + 1 < len(rebal_dates)
            else returns.index[-1]
        )

        for t in score.index:
            sector = sector_map.get(t, "UNKNOWN")
            side = "neutral"
            if t in longs:
                side = "long"
            elif t in shorts:
                side = "short"

            rows.append(
                {
                    "rebalance_date": dt,
                    "hold_start": hold_start,
                    "hold_end": hold_end,
                    "lookback_start": lookback_start,
                    "lookback_end": lookback_end,
                    "ticker": t,
                    "sector": sector,
                    "beta": betas.get(t, np.nan),
                    "vol_total": total_vols.get(t, np.nan),
                    "vol_residual": resid_vols.get(t, np.nan),
                    "vol_used": vols.get(t, np.nan),
                    "beta_winsor": betas_w.get(t, np.nan),
                    "vol_winsor": vols_w.get(t, np.nan),
                    "beta_z": beta_z.get(t, np.nan),
                    "vol_z": vol_z.get(t, np.nan),
                    "score": score.get(t, np.nan),
                    "rank": rank.get(t, np.nan),
                    "n_obs": counts.get(t, np.nan),
                    "side": side,
                    "weight": weights.get(t, 0.0),
                    "universe_size": len(score),
                    "long_n": len(longs),
                    "short_n": len(shorts),
                }
            )

    if not rows:
        raise RuntimeError("No rows generated. Check inputs.")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
