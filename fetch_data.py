"""Optional public-data refresh. Saved original results use a separately fingerprinted snapshot."""
from pathlib import Path
import argparse
import yfinance as yf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'data/raw/daily_returns.csv')
    args = parser.parse_args()
    prices = yf.download(['^GSPC','^FTSE','GLD','IEF'],start='2019-01-01',end='2026-03-06',
                         auto_adjust=True)['Close']
    expected = {'^GSPC','^FTSE','GLD','IEF'}
    if set(prices.columns) != expected or prices.isna().all().any():
        raise RuntimeError('Incomplete download; no output written')
    prices = prices.dropna().sort_index()
    returns = prices.pct_change(fill_method=None).dropna()
    if returns.empty:
        raise RuntimeError('No aligned returns; no output written')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Output exists. Choose --output with a new filename to preserve the snapshot.')
    returns.to_csv(args.output)
    print(f'Saved {len(returns)} aligned observations to {args.output}. Vendor revisions may change results.')


if __name__ == '__main__':
    main()
