# Low Risk Factor Demo

This repository lets you compute and visualize a **low‑risk stock factor** using
S&P 500 price data.  You don't need to be a programmer—just follow the steps
below.  By the end you'll have a CSV of factor weights and a set of charts you
can view in your web browser or image viewer.

> 🔁 The process is *deterministic*: running it multiple times produces the
> same output, so you can delete the results and rerun anytime.

---

## 📦 What’s included

- Python scripts organized as a package (`price_momentum/`)
- Shell helpers in `scripts/` for environment setup and pipeline execution
- `sp500.h5` – the raw historical data file used by the code
- `ticker_to_sector.json` – maps tickers to industries for sector neutrality
- `data/` – holds a cached copy of SPY prices downloaded by the scripts
- `requirements.txt` – list of Python dependencies
- A `.gitignore` that omits generated files from version control


## ✅ Step 1: open a terminal and navigate to the project

1. Open your terminal application (or the VS Code integrated terminal).
2. Change to the project folder.  For example:

   ```bash
   cd "/Users/lucaszelmanovits/Desktop/Quant Finance/Price Momentum"
   ```

   If you run `ls` now you should see `README.md`, `price_momentum/`,
   `scripts/`, etc.  This is the *project root* and all commands below assume
   you are here.

3. (Optional) confirm the directory with `pwd`.


## 🛠️ Step 2: set up the Python environment

A virtual environment keeps this project’s Python packages separate from
others on your computer.

### 🔧 Automatic setup (easiest)

```bash
chmod +x scripts/setup_env.sh          # do this once
tc
"./scripts/setup_env.sh"
```

- Creates `.venv/` if it's missing
- Installs everything from `requirements.txt` (NumPy, pandas, etc.)
- No need to manually activate the environment; the script uses it directly.

When it finishes the prompt may show `(.venv)` which indicates the env is
active.

### 🔩 Manual setup (just in case)

```bash
python3 -m venv .venv                   # create environment
# shellcheck source=/dev/null
. ".venv/bin/activate"                 # activate it
pip install -r requirements.txt         # install packages
```

> You only need to do this once.  If you ever come back later, just run
> `. ".venv/bin/activate"` to reactivate the environment.


## ▶️ Step 3: run the pipeline

There are two main stages: computing the factor inputs and generating the
visualizations.  A wrapper script runs them both in order.

```bash
chmod +x scripts/run_pipeline.sh        # one-time permission
"./scripts/run_pipeline.sh"            # run the full pipeline
```

You'll see progress messages printed to the terminal.  The script will
overwrite any existing files in `output_adj/` and leave you with fresh
results.

### Alternative: run stages individually

Activate the environment first (if it isn't already):

```bash
. ".venv/bin/activate"
```

Then execute each module:

```bash
python -m price_momentum.export_factor_inputs [options]
python -m price_momentum.generate_visuals [options]
```

Add `-h` to either command to see all available options (e.g. lookback window,
quantile, weighting).  The default settings are sensible for most use cases.


## 📁 Step 4: inspect the outputs

After running, these files will be created:

- `output_adj/factor_inputs.csv` – every rebalance date, ticker, score, and
  weight.
- `output_adj/visuals/` – a collection of PNG charts:
  - `long_vs_short_cumulative.png`
  - `rolling_sharpe.png`
  - `drawdown.png`
  - `signal_by_sector.png`
  - `decile_heatmap.png`
  - `beta_exposure.png`
  - `portfolio_evolution.png`

View them with any image viewer (e.g. double-click on macOS or use `open
output_adj/visuals/*.png`).

You can also open the CSV in Excel or with `head` to peek at the numbers.


## 🧹 Step 5: cleanup (optional)

To remove all results and start fresh:

```bash
rm -rf output_adj
```

This folder is recreated automatically the next time you run the pipeline.


## 📝 Additional notes

* The project path may contain spaces (e.g. “Quant Finance”).  The helper
  scripts properly quote paths so you don’t have to worry about this.
* The pipeline will emit a warning about OpenSSL (`NotOpenSSLWarning`) and a
  small pandas `FutureWarning`; these do not affect results.
* If you reinstall VS Code or move the repository, just rerun
  `./scripts/setup_env.sh` to rebuild the Python environment.


## 🗂️ Project structure recap

```
project-root/
├── README.md                 # this file
├── requirements.txt          # Python packages list
├── scripts/                  # shell helpers
│   ├── setup_env.sh
│   └── run_pipeline.sh
├── price_momentum/           # Python package with all the logic
│   ├── __init__.py
│   ├── low_risk_factor.py
│   ├── export_factor_inputs.py
│   ├── generate_visuals.py
│   ├── diagnose_low_risk.py
│   └── plot_portfolio.py
├── data/                     # caches (e.g. spy.csv)
├── sp500.h5                  # core dataset
├── ticker_to_sector.json     # sector mapping
└── output_adj/               # generated results (after running)
```
