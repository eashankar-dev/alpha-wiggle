import argparse
import os

import numpy as np
import pandas as pd

from .low_risk_factor import (
    build_scores,
    clean_prices,
    compute_beta_and_vol,
    load_prices_and_volume,
    load_prices_close,
    load_sector_map,
    load_sp500_tickers,
    load_spy_series,
    month_end_rebalance_dates,
    standardize_signals,
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate factor visuals")
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
    p.add_argument("--out-dir", default="output_adj/visuals")
    p.add_argument("--price-floor", type=float, default=0.1)
    p.add_argument("--price-ceil", type=float, default=1e5)
    p.add_argument("--return-cap", type=float, default=1.0)
    p.add_argument("--price-field", default=None)
    p.add_argument("--sector-map", default="ticker_to_sector.json")
    p.add_argument("--winsor-p", type=float, default=0.01)
    p.add_argument("--sector-neutral", action="store_true", default=True)
    p.add_argument("--no-sector-neutral", dest="sector_neutral", action="store_false")
    p.add_argument("--weighting", default="equal", choices=["equal", "value"])
    p.add_argument("--min-history-frac", type=float, default=0.8)
    p.add_argument("--min-dollar-vol", type=float, default=None)
    p.add_argument("--vol-type", default="residual", choices=["total", "residual"])
    p.add_argument("--bab-beta-neutral", action="store_true", default=True)
    p.add_argument("--bab-target-gross", type=float, default=2.0)
    return p.parse_args()


def _compute_leg_weights(longs, shorts, avg_dollar_vol, weighting, n):
    if weighting == "value":
        long_vals = avg_dollar_vol.reindex(longs).dropna()
        short_vals = avg_dollar_vol.reindex(shorts).dropna()
        long_w = long_vals / long_vals.sum() if long_vals.sum() > 0 else None
        short_w = short_vals / short_vals.sum() if short_vals.sum() > 0 else None
        return long_w, short_w

    if len(longs) > 0:
        long_w = pd.Series(1.0 / n, index=longs)
    else:
        long_w = None
    if len(shorts) > 0:
        short_w = pd.Series(1.0 / n, index=shorts)
    else:
        short_w = None
    return long_w, short_w


def main():
    args = parse_args()

    tickers = load_sp500_tickers(args.h5)
    sector_map = load_sector_map(args.sector_map)

    need_volume = args.weighting == "value" or args.min_dollar_vol is not None
    if need_volume:
        prices, volumes = load_prices_and_volume(args.h5, tickers, price_field=args.price_field)
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

    factor_returns = []
    long_returns = []
    short_returns = []
    beta_exposure = []
    decile_rows = []
    sector_scores_last = None

    min_obs = max(30, int(args.min_history_frac * args.lookback))

    for i, dt in enumerate(rebal_dates):
        pos = returns.index.get_loc(dt)
        end_pos = pos - args.skip
        start_pos = end_pos - args.lookback
        if start_pos < 0:
            continue

        window = returns.iloc[start_pos:end_pos]
        market_window = mkt_returns.iloc[start_pos:end_pos]

        betas, total_vols, resid_vols = compute_beta_and_vol(window, market_window, min_obs)
        vols = resid_vols if args.vol_type == "residual" else total_vols
        betas_w, vols_w, beta_z, vol_z = standardize_signals(
            betas, vols, sector_map, args.winsor_p, sector_neutral=args.sector_neutral
        )
        score = build_scores(beta_z, vol_z, args.method).dropna()

        # Liquidity filter
        avg_dollar_vol = None
        if args.min_dollar_vol is not None or args.weighting == "value":
            vol_window = volumes.iloc[start_pos:end_pos]
            price_window = prices.iloc[start_pos:end_pos]
            avg_dollar_vol = (price_window * vol_window).mean()
            if args.min_dollar_vol is not None:
                liquid = avg_dollar_vol[avg_dollar_vol >= args.min_dollar_vol].index
                score = score.loc[score.index.intersection(liquid)]

        if score.empty:
            continue

        # Save sector scores for the last rebalance for boxplot
        sector_scores_last = pd.DataFrame({
            "ticker": score.index,
            "sector": score.index.to_series().map(sector_map).fillna("UNKNOWN"),
            "score": score.values,
        })

        n = int(len(score) * args.quantile)
        if n < 1:
            continue

        longs = score.nsmallest(n).index
        shorts = score.nlargest(n).index

        # Weights for factor (long-short)
        weights = pd.Series(0.0, index=score.index)
        if args.side == "long_short":
            if args.weighting == "value":
                if avg_dollar_vol is None:
                    raise RuntimeError("value weighting requires volume data")
                long_vals = avg_dollar_vol.reindex(longs).dropna()
                short_vals = avg_dollar_vol.reindex(shorts).dropna()
                if long_vals.sum() > 0:
                    weights.loc[long_vals.index] = long_vals / long_vals.sum()
                if short_vals.sum() > 0:
                    weights.loc[short_vals.index] = -short_vals / short_vals.sum()
            else:
                weights.loc[longs] = 1.0 / n
                weights.loc[shorts] = -1.0 / n
        elif args.side == "long_only":
            weights.loc[longs] = 1.0 / n
        elif args.side == "short_only":
            weights.loc[shorts] = -1.0 / n

        # BAB beta-neutral scaling (for factor series + beta exposure)
        if args.bab_beta_neutral and args.side == "long_short":
            beta_slice = betas.reindex(weights.index)
            long_mask = weights > 0
            short_mask = weights < 0
            if long_mask.any() and short_mask.any():
                beta_long = (weights[long_mask] * beta_slice[long_mask]).sum()
                beta_short = (weights[short_mask] * beta_slice[short_mask]).sum()
                if beta_short != 0:
                    scale_short = -beta_long / beta_short
                    weights.loc[short_mask] = weights.loc[short_mask] * scale_short
                if args.bab_target_gross is not None:
                    gross = weights.abs().sum()
                    if gross > 0:
                        weights = weights * (args.bab_target_gross / gross)

        # Beta exposure series (per rebalance date)
        beta_exposure.append(
            (dt, (weights * betas.reindex(weights.index)).sum())
        )

        # Hold slice
        next_pos = (
            returns.index.get_loc(rebal_dates[i + 1])
            if i + 1 < len(rebal_dates)
            else len(returns.index) - 1
        )
        hold_slice = returns.iloc[pos + 1 : next_pos + 1]
        if hold_slice.empty:
            continue

        # Factor daily returns
        daily_factor = hold_slice[weights.index].mul(weights, axis=1).sum(axis=1, min_count=1)
        factor_returns.append(daily_factor)

        # Long vs Short leg returns (as if long each basket)
        long_w, short_w = _compute_leg_weights(longs, shorts, avg_dollar_vol, args.weighting, n)
        if long_w is not None:
            long_leg = hold_slice[long_w.index].mul(long_w, axis=1).sum(axis=1, min_count=1)
            long_returns.append(long_leg)
        if short_w is not None:
            short_leg = hold_slice[short_w.index].mul(short_w, axis=1).sum(axis=1, min_count=1)
            short_returns.append(short_leg)

        # Decile returns (monthly, compounded)
        ranks = score.rank(ascending=True, method="average")
        deciles = pd.qcut(ranks, 10, labels=False, duplicates="drop")
        for d in range(int(deciles.max()) + 1):
            tickers_d = deciles[deciles == d].index
            if len(tickers_d) == 0:
                continue
            if args.weighting == "value" and avg_dollar_vol is not None:
                w = avg_dollar_vol.reindex(tickers_d).dropna()
                if w.sum() == 0:
                    continue
                w = w / w.sum()
                ret = hold_slice[w.index].mul(w, axis=1).sum(axis=1, min_count=1)
            else:
                ret = hold_slice[tickers_d].mean(axis=1)
            period_return = (1 + ret).prod() - 1
            decile_rows.append({"rebalance_date": dt, "decile": d + 1, "return": period_return})

    if not factor_returns:
        raise RuntimeError("No factor returns computed.")

    factor = pd.concat(factor_returns).sort_index()
    long_leg = pd.concat(long_returns).sort_index() if long_returns else None
    short_leg = pd.concat(short_returns).sort_index() if short_returns else None

    # Rolling Sharpe (252‑day)
    roll = factor.rolling(252)
    rolling_sharpe = (roll.mean() / roll.std()) * np.sqrt(252)

    # Drawdown
    cum = (1 + factor).cumprod()
    drawdown = cum / cum.cummax() - 1

    # Beta exposure series
    beta_df = pd.DataFrame(beta_exposure, columns=["date", "beta"]).set_index("date")

    # Heatmap data
    decile_df = pd.DataFrame(decile_rows)
    heatmap = None
    if not decile_df.empty:
        heatmap = decile_df.pivot(index="decile", columns="rebalance_date", values="return")

    # Plotting
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Long vs Short cumulative
    plt.figure(figsize=(10, 5))
    if long_leg is not None:
        plt.plot((1 + long_leg).cumprod(), label="Long Leg", color="#1f77b4")
    if short_leg is not None:
        plt.plot((1 + short_leg).cumprod(), label="Short Leg", color="#d62728")
    plt.title("Long vs Short Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "long_vs_short_cumulative.png"), dpi=150)

    # 2) Rolling Sharpe
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_sharpe, color="#2ca02c")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Rolling 12‑Month Sharpe")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "rolling_sharpe.png"), dpi=150)

    # 3) Drawdown
    plt.figure(figsize=(10, 4))
    plt.plot(drawdown, color="#9467bd")
    plt.title("Drawdown Curve")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "drawdown.png"), dpi=150)

    # 4) Signal distribution by sector
    if sector_scores_last is not None:
        plt.figure(figsize=(12, 5))
        sectors = sector_scores_last["sector"].unique()
        data = [
            sector_scores_last.loc[sector_scores_last["sector"] == s, "score"]
            for s in sectors
        ]
        plt.boxplot(data, labels=sectors, vert=True, showfliers=False)
        plt.title("Signal Distribution by Sector")
        plt.ylabel("Combined Z‑Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "signal_by_sector.png"), dpi=150)

    # 5) Beta exposure over time
    plt.figure(figsize=(10, 4))
    plt.step(beta_df.index, beta_df["beta"], where="post", color="#ff7f0e")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Portfolio Beta Exposure Over Time")
    plt.xlabel("Date")
    plt.ylabel("Beta")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "beta_exposure.png"), dpi=150)

    # 6) Heatmap of decile returns
    if heatmap is not None:
        plt.figure(figsize=(12, 5))
        plt.imshow(heatmap, aspect="auto", cmap="RdYlGn", origin="lower")
        plt.title("Heatmap of Decile Returns (Next‑Period)")
        plt.xlabel("Rebalance Date")
        plt.ylabel("Decile (1=Low Risk)")
        plt.colorbar(label="Return")
        # show fewer x‑ticks for readability
        xticks = np.linspace(0, heatmap.shape[1] - 1, 8).astype(int)
        plt.xticks(xticks, [heatmap.columns[i].strftime("%Y") for i in xticks])
        plt.yticks(range(heatmap.shape[0]), heatmap.index)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "decile_heatmap.png"), dpi=150)

    print(f"wrote visuals to: {args.out_dir}")


if __name__ == "__main__":
    main()
