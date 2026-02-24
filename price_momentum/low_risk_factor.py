import argparse
import json
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import yfinance as yf


def _decode_list(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def load_sector_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("sector map must be a dict of ticker -> sector")
    return data


def load_sp500_tickers(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        g = f["sp500_tickers"]
        data = bytes(g["block0_values"][0].tolist())
        arr = pickle.loads(data)

    tickers = []
    for row in arr:
        val = row[0] if isinstance(row, (list, tuple, np.ndarray)) else row
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        tickers.append(str(val))
    return tickers


def winsorize_series(s, p):
    if p is None or p <= 0:
        return s
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def zscore_series(s):
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - s.mean()) / sd


def zscore_by_group(s, sector_map):
    labels = s.index.to_series().map(sector_map).fillna("UNKNOWN")

    def _z(x):
        sd = x.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return x * 0
        return (x - x.mean()) / sd

    return s.groupby(labels).transform(_z)


def standardize_signals(betas, vols, sector_map, winsor_p, sector_neutral=True):
    betas_w = winsorize_series(betas, winsor_p)
    vols_w = winsorize_series(vols, winsor_p)
    if sector_neutral:
        beta_z = zscore_by_group(betas_w, sector_map)
        vol_z = zscore_by_group(vols_w, sector_map)
    else:
        beta_z = zscore_series(betas_w)
        vol_z = zscore_series(vols_w)
    return betas_w, vols_w, beta_z, vol_z


def _select_price_column(df, price_field=None):
    cols = {c.lower(): c for c in df.columns}
    if price_field:
        key = price_field.lower()
        if key in cols:
            return cols[key]
    # prefer adjusted close if available
    for key in ["adjusted_close", "adj close", "adj_close", "adjusted close"]:
        if key in cols:
            return cols[key]
    for key in ["close"]:
        if key in cols:
            return cols[key]
    return None


def _select_volume_column(df):
    cols = {c.lower(): c for c in df.columns}
    for key in ["volume", "vol"]:
        if key in cols:
            return cols[key]
    return None


def load_prices_close(h5_path: str, tickers, price_field=None):
    prices = {}
    with h5py.File(h5_path, "r") as f:
        price_groups = {k for k in f.keys() if k.startswith("prices_")}
        for t in tickers:
            key = f"prices_{t}"
            if key not in price_groups:
                continue
            g = f[key]

            dates = pd.to_datetime(g["axis1"][:], unit="ns", utc=True).tz_convert(None)

            b0_items = _decode_list(g["block0_items"][:])
            b0_values = g["block0_values"][:]
            b1_items = _decode_list(g["block1_items"][:])
            b1_values = g["block1_values"][:]

            data = np.hstack([b0_values, b1_values])
            cols = b0_items + b1_items

            df = pd.DataFrame(data, index=dates, columns=cols)
            price_col = _select_price_column(df, price_field=price_field)
            if price_col is None:
                continue
            prices[t] = df[price_col].astype(float)

    prices_df = pd.DataFrame(prices).sort_index()
    return prices_df


def load_prices_and_volume(h5_path: str, tickers, price_field=None):
    prices = {}
    volumes = {}
    with h5py.File(h5_path, "r") as f:
        price_groups = {k for k in f.keys() if k.startswith("prices_")}
        for t in tickers:
            key = f"prices_{t}"
            if key not in price_groups:
                continue
            g = f[key]

            dates = pd.to_datetime(g["axis1"][:], unit="ns", utc=True).tz_convert(None)

            b0_items = _decode_list(g["block0_items"][:])
            b0_values = g["block0_values"][:]
            b1_items = _decode_list(g["block1_items"][:])
            b1_values = g["block1_values"][:]

            data = np.hstack([b0_values, b1_values])
            cols = b0_items + b1_items

            df = pd.DataFrame(data, index=dates, columns=cols)
            price_col = _select_price_column(df, price_field=price_field)
            vol_col = _select_volume_column(df)
            if price_col is None or vol_col is None:
                continue
            prices[t] = df[price_col].astype(float)
            volumes[t] = df[vol_col].astype(float)

    prices_df = pd.DataFrame(prices).sort_index()
    volumes_df = pd.DataFrame(volumes).sort_index()
    return prices_df, volumes_df


def load_spy_series(start, end, cache_path):
    if os.path.exists(cache_path):
        spy = pd.read_csv(cache_path)
        if "Date" in spy.columns:
            spy["Date"] = pd.to_datetime(spy["Date"])
            spy = spy.set_index("Date")
            if "Close" in spy.columns:
                return spy["Close"]
            if "Adj Close" in spy.columns:
                return spy["Adj Close"]

        # handle yfinance-style CSV with extra header rows
        raw = pd.read_csv(cache_path, header=None)
        date_rows = raw.index[raw.iloc[:, 0].astype(str) == "Date"]
        if len(date_rows) > 0:
            start_row = date_rows[0] + 1
            data = raw.iloc[start_row:].copy()
            data.columns = ["Date", "Close"]
            data["Date"] = pd.to_datetime(data["Date"])
            data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
            return data.set_index("Date")["Close"].dropna()

        raise ValueError("SPY cache file format not recognized")

    spy = yf.download(
        "SPY",
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if spy.empty:
        raise RuntimeError("SPY download returned no data")
    spy = spy[["Close"]]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    spy.to_csv(cache_path)
    return spy["Close"]


def month_end_rebalance_dates(index):
    s = index.to_series()
    # pandas >= 3.0 uses "ME" for month-end
    try:
        return s.groupby(pd.Grouper(freq="ME")).last().dropna().values
    except Exception:
        return s.groupby(pd.Grouper(freq="M")).last().dropna().values


def compute_beta_and_vol(window_returns, market_window, min_obs):
    betas = {}
    total_vols = {}
    resid_vols = {}

    m = market_window.values
    for ticker in window_returns.columns:
        s = window_returns[ticker].values
        mask = ~np.isnan(s) & ~np.isnan(m)
        if mask.sum() < min_obs:
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

    return pd.Series(betas), pd.Series(total_vols), pd.Series(resid_vols)


def build_scores(beta_signal, vol_signal, method):
    method = method.lower()
    if method == "beta":
        score = beta_signal
    elif method == "vol":
        score = vol_signal
    elif method == "combined":
        score = (beta_signal + vol_signal) / 2.0
    else:
        raise ValueError("method must be one of: beta, vol, combined")

    return score


def clean_prices(prices, price_floor=None, price_ceil=None):
    if price_floor is not None:
        prices = prices.where(prices >= price_floor)
    if price_ceil is not None:
        prices = prices.where(prices <= price_ceil)
    return prices


def run_low_risk_factor(
    h5_path,
    lookback=252,
    skip=21,
    quantile=0.2,
    method="combined",
    side="long_short",
    spy_cache="data/spy.csv",
    output_dir="output",
    price_floor=0.1,
    price_ceil=1e5,
    return_cap=1.0,
    price_field=None,
    sector_map_path="ticker_to_sector.json",
    winsor_p=0.01,
    sector_neutral=True,
    weighting="equal",
    min_history_frac=0.8,
    min_dollar_vol=None,
    vol_type="total",
    bab_beta_neutral=False,
    bab_target_gross=2.0,
):
    tickers = load_sp500_tickers(h5_path)
    sector_map = load_sector_map(sector_map_path)

    need_volume = weighting == "value" or min_dollar_vol is not None
    if need_volume:
        prices, volumes = load_prices_and_volume(
            h5_path, tickers, price_field=price_field
        )
    else:
        prices = load_prices_close(h5_path, tickers, price_field=price_field)
        volumes = None

    prices = clean_prices(prices, price_floor=price_floor, price_ceil=price_ceil)
    returns = prices.pct_change().dropna(how="all")
    if return_cap is not None:
        returns = returns.where(returns.abs() <= return_cap)

    start = returns.index.min().date().isoformat()
    end = (returns.index.max() + pd.Timedelta(days=1)).date().isoformat()
    spy = load_spy_series(start, end, spy_cache)
    mkt_returns = spy.pct_change().reindex(returns.index)

    common = returns.index.intersection(mkt_returns.index)
    returns = returns.loc[common]
    mkt_returns = mkt_returns.loc[common]

    rebal_dates = month_end_rebalance_dates(returns.index)

    factor_returns = []
    turnover = []
    prev_weights = None

    for i, dt in enumerate(rebal_dates):
        pos = returns.index.get_loc(dt)
        end_pos = pos - skip
        start_pos = end_pos - lookback
        if start_pos < 0:
            continue

        window = returns.iloc[start_pos:end_pos]
        market_window = mkt_returns.iloc[start_pos:end_pos]

        min_obs = max(30, int(min_history_frac * lookback))
        betas, total_vols, resid_vols = compute_beta_and_vol(
            window, market_window, min_obs
        )
        vols = resid_vols if vol_type == "residual" else total_vols
        betas_w, vols_w, beta_z, vol_z = standardize_signals(
            betas, vols, sector_map, winsor_p, sector_neutral=sector_neutral
        )
        score = build_scores(beta_z, vol_z, method)
        score = score.dropna()

        # Liquidity filter based on average dollar volume in the lookback window
        avg_dollar_vol = None
        if min_dollar_vol is not None or weighting == "value":
            vol_window = volumes.iloc[start_pos:end_pos]
            price_window = prices.iloc[start_pos:end_pos]
            avg_dollar_vol = (price_window * vol_window).mean()
            if min_dollar_vol is not None:
                liquid = avg_dollar_vol[avg_dollar_vol >= min_dollar_vol].index
                score = score.loc[score.index.intersection(liquid)]

        if score.empty:
            continue

        n = int(len(score) * quantile)
        if n < 1:
            continue

        longs = score.nsmallest(n).index
        shorts = score.nlargest(n).index

        weights = pd.Series(0.0, index=score.index)
        if weighting == "value":
            if avg_dollar_vol is None:
                raise RuntimeError("value weighting requires volume data")
            long_vals = avg_dollar_vol.reindex(longs).dropna()
            short_vals = avg_dollar_vol.reindex(shorts).dropna()
            if side == "long_short":
                if long_vals.sum() > 0:
                    weights.loc[long_vals.index] = long_vals / long_vals.sum()
                if short_vals.sum() > 0:
                    weights.loc[short_vals.index] = -short_vals / short_vals.sum()
            elif side == "long_only":
                if long_vals.sum() > 0:
                    weights.loc[long_vals.index] = long_vals / long_vals.sum()
            elif side == "short_only":
                if short_vals.sum() > 0:
                    weights.loc[short_vals.index] = -short_vals / short_vals.sum()
            else:
                raise ValueError("side must be one of: long_short, long_only, short_only")
        else:
            if side == "long_short":
                weights.loc[longs] = 1.0 / n
                weights.loc[shorts] = -1.0 / n
            elif side == "long_only":
                weights.loc[longs] = 1.0 / n
            elif side == "short_only":
                weights.loc[shorts] = -1.0 / n
            else:
                raise ValueError("side must be one of: long_short, long_only, short_only")
        # optional BAB beta-neutral scaling (long-short only)
        if bab_beta_neutral and side == "long_short":
            beta_slice = betas.reindex(weights.index)
            long_mask = weights > 0
            short_mask = weights < 0
            if long_mask.any() and short_mask.any():
                beta_long = (weights[long_mask] * beta_slice[long_mask]).sum()
                beta_short = (weights[short_mask] * beta_slice[short_mask]).sum()
                if beta_short != 0:
                    scale_short = -beta_long / beta_short
                    weights.loc[short_mask] = weights.loc[short_mask] * scale_short
                if bab_target_gross is not None:
                    gross = weights.abs().sum()
                    if gross > 0:
                        weights = weights * (bab_target_gross / gross)

        if prev_weights is None:
            turnover.append(np.nan)
        else:
            w = weights.reindex(prev_weights.index).fillna(0.0)
            prev = prev_weights.reindex(weights.index).fillna(0.0)
            turnover.append((w - prev).abs().sum())
        prev_weights = weights

        next_pos = (
            returns.index.get_loc(rebal_dates[i + 1])
            if i + 1 < len(rebal_dates)
            else len(returns.index) - 1
        )
        hold_slice = returns.iloc[pos + 1 : next_pos + 1]
        if hold_slice.empty:
            continue

        daily = hold_slice[weights.index].mul(weights, axis=1)
        port = daily.sum(axis=1, min_count=1)
        factor_returns.append(port)

    if not factor_returns:
        raise RuntimeError("No factor returns computed. Check lookback/skip/data.")

    factor = pd.concat(factor_returns).sort_index()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "low_risk_factor_returns.csv")
    factor.to_csv(out_path, header=["factor_return"])

    summary = {}
    summary["start"] = factor.index.min().date().isoformat()
    summary["end"] = factor.index.max().date().isoformat()
    summary["days"] = len(factor)
    summary["mean_daily"] = float(factor.mean())
    summary["vol_daily"] = float(factor.std())
    summary["sharpe_daily"] = float(summary["mean_daily"] / summary["vol_daily"])

    summary_path = os.path.join(output_dir, "low_risk_factor_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    return out_path, summary_path


def parse_args():
    p = argparse.ArgumentParser(description="Low risk factor from sp500.h5")
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
    p.add_argument("--long-short", action="store_true", default=False)
    p.add_argument("--long-only", action="store_true", default=False)
    p.add_argument("--short-only", action="store_true", default=False)
    p.add_argument("--spy-cache", default="data/spy.csv")
    p.add_argument("--output-dir", default="output")
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
    if args.long_short:
        args.side = "long_short"
    if args.long_only:
        args.side = "long_only"
    if args.short_only:
        args.side = "short_only"
    out_path, summary_path = run_low_risk_factor(
        h5_path=args.h5,
        lookback=args.lookback,
        skip=args.skip,
        quantile=args.quantile,
        method=args.method,
        side=args.side,
        spy_cache=args.spy_cache,
        output_dir=args.output_dir,
        price_floor=args.price_floor,
        price_ceil=args.price_ceil,
        return_cap=args.return_cap,
        price_field=args.price_field,
        sector_map_path=args.sector_map,
        winsor_p=args.winsor_p,
        sector_neutral=args.sector_neutral,
        weighting=args.weighting,
        min_history_frac=args.min_history_frac,
        min_dollar_vol=args.min_dollar_vol,
        vol_type=args.vol_type,
        bab_beta_neutral=args.bab_beta_neutral,
        bab_target_gross=args.bab_target_gross,
    )
    print(f"wrote: {out_path}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
