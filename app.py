"""Streamlit interface for the rolling VaR and Expected Shortfall backtests in risk.py.

Every number shown here is produced by the functions in risk.py that the test suite
covers. Nothing is recomputed in this file, so the browser interface and the command
line cannot quietly disagree with each other.
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import risk

CONFIDENCES = [0.95, 0.99]
REPO = "https://github.com/ajayworks/market-risk-analytics"

st.set_page_config(page_title="Market risk backtesting", layout="wide")


@st.cache_data(show_spinner=False)
def simulated_returns(observations=1500, seed=7):
    """Four correlated series driven by a GARCH-like common factor.

    Volatility clustering is built in deliberately: a demonstration that fed the
    models well-behaved noise would flatter them and the independence test would
    have nothing to find. This is simulated data and is not evidence about any
    real market.
    """
    generator = np.random.default_rng(seed)
    omega, alpha, beta = 1e-6, 0.09, 0.90
    variance = omega/(1-alpha-beta)
    factor = np.empty(observations)
    for position in range(observations):
        shock = generator.standard_normal()*np.sqrt(variance)
        factor[position] = shock
        variance = omega + alpha*shock**2 + beta*variance
    loadings = np.array([0.35, 0.25, 0.95, 1.05])
    frame = pd.DataFrame(
        factor[:, None]*loadings + generator.standard_normal((observations, 4))*0.004,
        index=pd.bdate_range("2019-01-02", periods=observations),
        columns=["Gold proxy", "Bond proxy", "UK equity proxy", "US equity proxy"])
    # Routed through the same validator as an uploaded file, so the demonstration
    # path cannot bypass a check that user data has to pass.
    buffer = io.StringIO()
    frame.to_csv(buffer)
    buffer.seek(0)
    return risk.load_returns(buffer)


@st.cache_data(show_spinner=False)
def uploaded_returns(payload):
    return risk.load_returns(io.BytesIO(payload))


@st.cache_data(show_spinner=False)
def backtest(frame, window):
    series, weights = risk.portfolio_returns(frame)
    forecasts = pd.concat([risk.rolling_forecasts(series, window, c) for c in CONFIDENCES],
                          ignore_index=True)
    rows = []
    for (method, confidence), group in forecasts.groupby(["method", "confidence"]):
        unconditional = risk.coverage_test(group.breach, confidence)
        independence = risk.independence_test(group.breach)
        rows.append({"method": method, "confidence": confidence, **unconditional,
                     **independence, **risk.conditional_coverage(unconditional, independence)})
    contributions, volatility = risk.risk_contributions(frame, weights)
    return forecasts, pd.DataFrame(rows), contributions, volatility


st.sidebar.header("Return data")
source = st.sidebar.radio("Source", ["Simulated demonstration data", "Upload a CSV"])
upload = None
if source == "Upload a CSV":
    upload = st.sidebar.file_uploader(
        "Dates in the first column, one column per asset, simple returns", type="csv")
window = st.sidebar.slider("Estimation window (prior observations)", 100, 500, 250, 25)
st.sidebar.caption("Every forecast is estimated only from observations strictly before "
                   "the one it predicts. Equal weights, no transaction costs.")

st.title("Rolling VaR and Expected Shortfall backtesting")
st.write("Next-observation Value at Risk and Expected Shortfall forecasts under historical "
         "and Gaussian methods, evaluated with Kupiec unconditional coverage, Christoffersen "
         "independence and conditional coverage tests of the VaR exceptions.")

try:
    if source == "Upload a CSV":
        if upload is None:
            st.info("Upload a return file, or keep the simulated data to see the full workflow.")
            st.stop()
        frame = uploaded_returns(upload.getvalue())
    else:
        frame = simulated_returns()
except ValueError as error:
    st.error(f"The return file was rejected before any model ran: {error}")
    st.stop()

if len(frame) <= window:
    st.error(f"{len(frame):,} observations cannot support a {window}-observation estimation "
             "window. Reduce the window in the sidebar.")
    st.stop()

forecasts, summary, contributions, volatility = backtest(frame, window)

first, second, third = st.columns(3)
first.metric("Return observations", f"{len(frame):,}")
second.metric("Assets", frame.shape[1])
third.metric("Forecasts per model", f"{int(summary.observations.max()):,}")

st.subheader("Forecast coverage")
st.dataframe(pd.DataFrame({
    "Method": summary.method,
    "Confidence": summary.confidence.map("{:.0%}".format),
    "Forecasts": summary.observations,
    "Breaches": summary.breaches,
    "Expected": summary.expected_breaches.round(1),
    "Kupiec p": summary.kupiec_p_value.map(risk.format_p_value),
    "Independence p": summary.christoffersen_ind_p_value.map(risk.format_p_value),
    "Conditional coverage p": summary.conditional_coverage_p_value.map(risk.format_p_value),
}), hide_index=True)
st.caption("Kupiec asks only how many breaches occurred. Christoffersen asks whether a breach "
           "makes the next one more likely. Conditional coverage combines the two likelihood "
           "ratios as a chi-squared statistic with two degrees of freedom. Non-rejection is "
           "not validation.")

st.subheader("Breach clustering")
st.dataframe(pd.DataFrame({
    "Method": summary.method,
    "Confidence": summary.confidence.map("{:.0%}".format),
    "Breach rate after a calm observation": summary.breach_rate_after_calm.map("{:.2%}".format),
    "Breach rate after a breach": summary.breach_rate_after_breach.map("{:.2%}".format),
}), hide_index=True)
st.caption("A model that fails in bursts is more dangerous than one that fails at random, and "
           "an unconditional test cannot tell them apart.")

st.subheader("Forecasts against realised losses")
left, right = st.columns(2)
method = left.selectbox("Method", sorted(forecasts.method.unique()))
confidence = right.selectbox("Confidence", CONFIDENCES, index=1, format_func="{:.0%}".format)
subset = forecasts[(forecasts.method == method) & (forecasts.confidence == confidence)]
figure, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
axis.plot(subset.date, subset.loss*100, color="#9ba7b4", linewidth=.6, label="Realised loss")
axis.plot(subset.date, subset["var"]*100, color="#145b79", linewidth=1,
          label=f"{confidence:.0%} VaR forecast")
breaches = subset[subset.breach]
axis.scatter(breaches.date, breaches.loss*100, color="#b44530", s=13, zorder=3, label="Breach")
axis.set(ylabel="Loss (%)")
axis.legend(loc="upper left", frameon=False)
st.pyplot(figure)
plt.close(figure)

st.subheader("Full-sample risk contributions")
st.dataframe(pd.DataFrame({
    "Asset": contributions.asset,
    "Weight": contributions.weight.map("{:.1%}".format),
    "Annualised volatility contribution":
        contributions.annualised_component_volatility.map("{:.2%}".format),
}), hide_index=True)
st.caption(f"Euler decomposition of the full-sample annualised portfolio volatility of "
           f"{volatility:.2%}. Descriptive, and excluded from the forecast evaluation above.")

with st.expander("Scope and limitations"):
    st.markdown(
        "- None of these tests establishes Expected Shortfall adequacy. At 99% the historical "
        "ES estimate averages only the three largest losses in each 250-observation window.\n"
        "- Equal weights are reset each observation and there are no transaction costs.\n"
        "- Risk contributions and portfolio volatility use the full sample, so they see data "
        "the forecasts were never allowed to see.\n"
        "- The simulated series are generated from a GARCH-like process for demonstration. "
        f"The published results in the repository use real market data: {REPO}")

st.caption(f"Source, test suite and published results: {REPO}")
