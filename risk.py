"""Transparent portfolio-risk analysis with rolling, strictly prior-data forecasts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binomtest, chi2, norm

ROOT = Path(__file__).resolve().parent


def load_returns(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.empty or frame.shape[1] < 1:
        raise ValueError("Return data must not be empty")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError("Dates must be valid")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Dates must be unique and increasing")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Returns must be numeric and finite; no silent dropping or filling")
    if (frame <= -1).any().any():
        raise ValueError("Simple returns must exceed -100%")
    return frame


def portfolio_returns(frame, weights=None):
    weights = np.ones(frame.shape[1]) / frame.shape[1] if weights is None else np.asarray(weights)
    if len(weights) != frame.shape[1] or not np.isfinite(weights).all():
        raise ValueError("Provide one finite weight per asset")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("This experiment assumes long-only weights summing to one")
    return frame @ weights, weights


def historical_risk(sample, confidence):
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one")
    losses = -np.asarray(sample, dtype=float)
    var = float(np.quantile(losses, confidence))
    es = float(losses[losses >= var].mean())
    return var, es


def rolling_forecasts(series, window=250, confidence=0.99):
    if window < 2 or len(series) <= window:
        raise ValueError("Need a window of at least two and observations beyond it")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one")
    records = []
    for position in range(window, len(series)):
        # Critical invariant: the forecast for t cannot see observation t or later.
        sample = series.iloc[position-window:position].to_numpy()
        var_h, es_h = historical_risk(sample, confidence)
        mu, sigma = sample.mean(), sample.std(ddof=1)
        z = norm.ppf(confidence)
        loss = -float(series.iloc[position])
        for method, var, es in [
            ("Historical", var_h, es_h),
            ("Gaussian", -mu + sigma*z, -mu + sigma*norm.pdf(z)/(1-confidence))]:
            records.append({"date": series.index[position], "method": method,
                            "confidence": confidence, "train_start": series.index[position-window],
                            "train_end": series.index[position-1], "loss": loss,
                            "var": float(var), "es": float(es), "breach": bool(loss > var)})
    return pd.DataFrame(records)


def coverage_test(breaches, confidence):
    b = np.asarray(breaches, dtype=bool)
    n, count = len(b), int(b.sum())
    if n == 0 or not 0 < confidence < 1:
        raise ValueError("Need observations and valid confidence")
    p, observed = 1-confidence, count/n
    null = xlogy(count, p) + xlogy(n-count, 1-p)
    fitted = xlogy(count, observed) + xlogy(n-count, 1-observed)
    lr = max(0.0, float(-2*(null-fitted)))
    return {"observations": n, "breaches": count, "expected_breaches": n*p,
            "breach_rate": observed, "kupiec_lr": lr,
            "kupiec_p_value": float(chi2.sf(lr, 1)),
            "exact_binomial_p_value": float(binomtest(count, n, p).pvalue),
            "interpretation": "Reject coverage at 5%" if chi2.sf(lr, 1) < .05
                              else "Coverage not rejected at 5% (not proof of validity)"}

def independence_test(breaches):
    """Christoffersen independence: does a breach make the next breach more likely?

    Kupiec tests only how many breaches occurred, never their arrangement. A model
    that fails in bursts is far more dangerous than one that fails at random, and
    the two are indistinguishable to an unconditional test.
    """
    b = np.asarray(breaches, dtype=bool)
    if len(b) < 2:
        raise ValueError("Need at least two observations to test independence")
    previous, current = b[:-1], b[1:]
    n00 = int((~previous & ~current).sum())
    n01 = int((~previous & current).sum())
    n10 = int((previous & ~current).sum())
    n11 = int((previous & current).sum())
    # Guard the degenerate rows: with no breaches at all the second row is empty.
    pi01 = n01/(n00+n01) if (n00+n01) else 0.0
    pi11 = n11/(n10+n11) if (n10+n11) else 0.0
    pi = (n01+n11)/(n00+n01+n10+n11)
    null = xlogy(n01+n11, pi) + xlogy(n00+n10, 1-pi)
    fitted = (xlogy(n01, pi01) + xlogy(n00, 1-pi01)
              + xlogy(n11, pi11) + xlogy(n10, 1-pi11))
    lr = max(0.0, float(-2*(null-fitted)))
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "breach_rate_after_calm": pi01, "breach_rate_after_breach": pi11,
            "christoffersen_ind_lr": lr,
            "christoffersen_ind_p_value": float(chi2.sf(lr, 1))}

def format_p_value(value):
    """A p-value of 4e-09 printed as 0.000000 reads as exactly zero. It is not."""
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.1e}"

def risk_contributions(frame, weights):
    covariance = frame.cov().to_numpy()*252
    vol = float(np.sqrt(weights @ covariance @ weights))
    if vol == 0:
        contributions = np.zeros_like(weights)
    else:
        contributions = weights*(covariance @ weights)/vol
    return pd.DataFrame({"asset": frame.columns, "weight": weights,
                         "annualised_component_volatility": contributions}), vol


def stress_scenarios(frame, weights):
    # Illustrative return shocks, not forecasts or regulatory stress specifications.
    shocks = {
        "Equity sell-off": {"GLD": .03, "IEF": .02, "^FTSE": -.15, "^GSPC": -.20},
        "Rates and inflation shock": {"GLD": -.05, "IEF": -.08, "^FTSE": -.10, "^GSPC": -.12},
        "Broad liquidation": {"GLD": -.10, "IEF": -.04, "^FTSE": -.18, "^GSPC": -.20}}
    if set(frame.columns) != {"GLD", "IEF", "^FTSE", "^GSPC"}:
        return []
    return [{"scenario": name, "asset_return_shocks": values,
             "portfolio_return": float(sum(weights[i]*values[c] for i,c in enumerate(frame.columns)))}
            for name, values in shocks.items()]


def run(data_path, output, window=250, seed=42):
    frame = load_returns(data_path)
    series, weights = portfolio_returns(frame)
    forecasts = pd.concat([rolling_forecasts(series, window, c) for c in [.95,.99]], ignore_index=True)
    summaries = []
    for (method, confidence), group in forecasts.groupby(["method", "confidence"]):
        unconditional = coverage_test(group.breach, confidence)
        independence = independence_test(group.breach)
        # Conditional coverage: the two likelihood ratios are additive, chi-squared with 2 df.
        cc_lr = unconditional["kupiec_lr"] + independence["christoffersen_ind_lr"]
        summaries.append({"method": method, "confidence": confidence,
                          **unconditional, **independence,
                          "conditional_coverage_lr": cc_lr,
                          "conditional_coverage_p_value": float(chi2.sf(cc_lr, 2))})
    contributions, vol = risk_contributions(frame, weights)
    rng = np.random.default_rng(seed)
    # method="cholesky" is deterministic across LAPACK builds. The default "svd" is not:
    # singular vectors are defined only up to sign, so different builds legitimately
    # return different decompositions and the same seed yields different draws.
    simulation = rng.multivariate_normal(frame.mean().to_numpy(), frame.cov().to_numpy(),
                                         10000, method="cholesky") @ weights
    mc = {str(c): dict(zip(["var", "es"], historical_risk(simulation,c))) for c in [.95,.99]}
    scenarios = stress_scenarios(frame, weights)
    versions = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    manifest = {"input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "python_version": platform.python_version(),
                "library_versions": versions,
                "start": str(frame.index.min().date()), "end": str(frame.index.max().date()),
                "return_observations": len(frame), "assets": list(frame.columns),
                "weights": weights.tolist(), "window": window, "seed": seed,
                "forecast_horizon": "next aligned return observation",
                "annualisation_factor": 252,
                "full_sample_annualised_portfolio_volatility": vol,
                "monte_carlo": {"scenarios": 10000, "scope": "full-sample descriptive estimate, not a backtest", "results": mc}}
    output.mkdir(parents=True,exist_ok=True)
    forecasts.to_csv(output/'rolling_forecasts.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'coverage_summary.csv',index=False)
    contributions.to_csv(output/'risk_contributions.csv',index=False)
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'stress_scenarios.json').write_text(json.dumps(scenarios,indent=2))
    plot_results(forecasts, contributions, output)
    lines = ['# Market Risk Analytics: results', '',
             f"Input: {manifest['start']} to {manifest['end']}; {len(frame):,} return observations; {window} prior observations per forecast.", '',
             '## Forecast coverage', '',
             '| Method | Confidence | Forecasts | Breaches | Expected | Kupiec p | Independence p | Conditional coverage p |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['method']} | {s['confidence']:.0%} | {s['observations']} | {s['breaches']} | "
                     f"{s['expected_breaches']:.1f} | {format_p_value(s['kupiec_p_value'])} | "
                     f"{format_p_value(s['christoffersen_ind_p_value'])} | "
                     f"{format_p_value(s['conditional_coverage_p_value'])} |")
    lines += ['', '## Breach clustering', '',
              '| Method | Confidence | Breach rate after a calm observation | Breach rate after a breach |',
              '|---|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['method']} | {s['confidence']:.0%} | {s['breach_rate_after_calm']:.2%} | "
                     f"{s['breach_rate_after_breach']:.2%} |")
    lines += ['', 'Each forecast excludes the realised observation and all future data. Kupiec tests only how many breaches occurred; the Christoffersen independence test asks whether a breach makes the next breach more likely; conditional coverage combines the two as a chi-squared statistic with two degrees of freedom. Non-rejection is not validation, and none of these tests establishes Expected Shortfall adequacy: at 99% the historical ES estimate averages only the three largest losses in each 250-observation window.', '',
              '![Forecast diagnostics](risk_diagnostics.png)', '',
              '## Hypothetical stress scenarios', '', '| Scenario | Portfolio return |','|---|---:|']
    lines += [f"| {s['scenario']} | {s['portfolio_return']:.2%} |" for s in scenarios]
    lines += ['', 'Shocks are explicitly chosen illustrations, not historical estimates or regulatory scenarios. Exact per-asset assumptions are in stress_scenarios.json.', '',
              '## Scope', '',
              'Equal weights are reset each observation; no transaction costs. Returns are from the original common-date dataset. Cross-market holiday alignment can make an observation span more than one trading day. FTSE and US assets are used as normalised local-return exposures without FX conversion, so this is not a fully specified single-currency investable portfolio. Full-sample covariance, risk contributions and Monte Carlo estimates are descriptive and are excluded from forecast evaluation.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False))
    return manifest


def plot_results(forecasts, contributions, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    subset=forecasts[(forecasts.method=='Historical') & (forecasts.confidence==.99)]
    axes[0].plot(subset.date,subset.loss*100,color='#9ba7b4',linewidth=.6,label='Realised loss')
    axes[0].plot(subset.date,subset['var']*100,color='#145b79',linewidth=1,label='Prior-window 99% VaR')
    failures=subset[subset.breach]
    axes[0].scatter(failures.date,failures.loss*100,color='#b44530',s=13,label='VaR breach',zorder=3)
    axes[0].set(title='Historical VaR: next-observation forecasts',ylabel='Loss (%)')
    axes[0].legend(loc='upper right',frameon=False)
    axes[1].bar(contributions.asset,contributions.annualised_component_volatility*100,color='#145b79')
    axes[1].set(title='Full-sample risk contributions (descriptive)',ylabel='Annualised volatility contribution (pp)')
    fig.savefig(output/'risk_diagnostics.png',dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/raw/daily_returns.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--window',type=int,default=250)
    args=parser.parse_args()
    run(args.data,args.output,args.window)


if __name__=='__main__':
    main()
