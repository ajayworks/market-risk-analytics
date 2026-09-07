# Reproduction on a second environment

Output of `python risk.py` run on a different operating system, Python version and NumPy
version from the run that produced `../run_manifest.json`. Both runs used the same input
file; the SHA-256 recorded in each manifest is identical.

| | committed run | reproduction run |
|---|---|---|
| Operating system | macOS (arm64) | Linux (x86_64) |
| Python | 3.12.2 | 3.10.12 |
| NumPy | 2.4.0 | 2.2.6 |
| SciPy | 1.16.3 | 1.15.3 |

## What matches exactly

- `coverage_summary.csv` is **byte-identical**. Every breach count, transition count, Kupiec,
  Christoffersen independence and conditional coverage statistic reproduces exactly. Diff
  `coverage_summary_linux.csv` against `../coverage_summary.csv` to check.
- The input SHA-256 and the full-sample annualised portfolio volatility are identical.

## What does not match exactly

Monte Carlo estimates agree to fifteen significant figures but differ in the final one or two
bits at the 95% level:

```
0.95 VaR   macOS 0.009250798672132151   Linux 0.009250798672132153
0.95 ES    macOS 0.011606781689725355   Linux 0.011606781689725358
0.99 VaR   0.013206988284364271         identical
0.99 ES    0.015017720608509075         identical
```

This is floating-point summation order in the portfolio aggregation, not a difference in the
simulated draws. Pinning `method="cholesky"` removed the decomposition-level non-determinism
described in the main README; what remains is at the limit of double precision and cannot be
eliminated without fixing the reduction order.

Monte Carlo estimates are descriptive and are excluded from the forecast evaluation, so this
does not affect any reported finding.
