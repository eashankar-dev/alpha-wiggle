import argparse
import os

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Plot portfolio evolution")
    p.add_argument(
        "--returns",
        default="output_adj/resid_bab_long_short/low_risk_factor_returns.csv",
        help="CSV of daily factor returns",
    )
    p.add_argument(
        "--out",
        default="output_adj/portfolio_evolution.png",
        help="Output image path",
    )
    p.add_argument(
        "--title",
        default="Low Risk Portfolio Evolution",
        help="Plot title",
    )
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.returns, index_col=0, parse_dates=True)
    if df.shape[1] == 0:
        raise RuntimeError("returns file has no columns")

    ret = df.iloc[:, 0].astype(float)
    cum = (1 + ret).cumprod()

    # Lazy import so the script can be inspected without matplotlib installed
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(cum.index, cum.values, linewidth=1.5, color="#1f77b4")
    plt.title(args.title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
