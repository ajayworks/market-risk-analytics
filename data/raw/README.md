# Input data

`daily_returns.csv` is the original project's Yahoo Finance adjusted-price return snapshot: 1,767 observations, 3 January 2019 to 5 March 2026, for GLD, IEF, ^FTSE and ^GSPC. Its exact SHA-256 is recorded in `../../results/run_manifest.json`, alongside the Python and library versions of the run that produced the published figures.

`clean_prices.csv` is the adjusted-close price series it was derived from: 1,768 observations from 2 January 2019, one row longer because the first date is consumed by differencing. Simple returns are `pct_change` of those prices with no fill.

Neither file is under version control, so **a fresh clone will not contain them**. `python fetch_data.py` downloads a replacement after installing the requirements; it refuses to overwrite an existing snapshot, and `--output data/raw/refreshed_returns.csv` writes a separate file to pass to `risk.py --data`.

A fresh download will **not** reproduce the published figures. Vendor adjustments and calendar changes revise historical adjusted prices, so the SHA-256 above is a fingerprint of one specific snapshot rather than something a later download can be expected to match. No download was performed during this revision.

The common-date alignment drops rows with missing prices before calculating returns, so some observations span more than one market session. No FX conversion is applied: these are normalised local-return exposures, not a complete single-currency portfolio.