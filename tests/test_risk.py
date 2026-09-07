import json
import os
import pathlib
import tempfile
import unittest
import numpy as np
import pandas as pd
from risk import (rolling_forecasts, coverage_test, historical_risk, risk_contributions,
                  independence_test, load_returns, run)


class RiskValidationTests(unittest.TestCase):
    def setUp(self):
        self.series = pd.Series(np.random.default_rng(5).normal(0, .01, 80),
                                index=pd.bdate_range('2020-01-01',periods=80))

    def test_current_and_future_observations_cannot_change_forecast(self):
        original = rolling_forecasts(self.series,20)
        changed = self.series.copy()
        changed.iloc[40:] = .5
        revised = rolling_forecasts(changed,20)
        mask = original.date <= self.series.index[40]
        np.testing.assert_allclose(original.loc[mask,'var'],revised.loc[mask,'var'])
        np.testing.assert_allclose(original.loc[mask,'es'],revised.loc[mask,'es'])
        self.assertTrue((original.train_end < original.date).all())

    def test_coverage_handles_zero_and_all_breaches(self):
        for flags in [[False]*100,[True]*100]:
            result=coverage_test(flags,.99)
            self.assertTrue(np.isfinite(result['kupiec_lr']))
            self.assertTrue(0 <= result['kupiec_p_value'] <= 1)

    def test_expected_shortfall_exceeds_var(self):
        for c in [.95,.99]:
            var,es=historical_risk(self.series,c)
            self.assertGreaterEqual(es,var)

    def test_contributions_sum_to_total_volatility(self):
        frame=pd.DataFrame({'a':self.series,'b':self.series.shift(1).fillna(0)})
        contributions,vol=risk_contributions(frame,np.array([.5,.5]))
        self.assertAlmostEqual(contributions.annualised_component_volatility.sum(),vol)

    def test_no_forecast_without_training_history(self):
        with self.assertRaises(ValueError):
            rolling_forecasts(self.series,80)
    def test_gaussian_var_and_es_match_analytic_normal_values(self):
        # A sample of +/-sqrt(19/20) has mean exactly 0 and sample standard deviation
        # exactly 1, so the Gaussian forecast must reproduce the standard normal
        # quantile and tail expectation. A sign error would not survive this.
        c = np.sqrt(19/20)
        series = pd.Series([-c]*10 + [c]*10 + [0.0],
                           index=pd.bdate_range('2020-01-01', periods=21))
        forecasts = rolling_forecasts(series, 20, .99)
        gaussian = forecasts[forecasts.method == 'Gaussian'].iloc[0]
        self.assertAlmostEqual(gaussian['var'], 2.3263478740408408, places=9)
        self.assertAlmostEqual(gaussian['es'], 2.6652142203458080, places=9)

    def test_independence_flags_clustering_and_clears_independent_breaches(self):
        clustered = [True]*10 + [False]*990
        self.assertLess(independence_test(clustered)['christoffersen_ind_p_value'], .001)
        independent = np.random.default_rng(11).random(1000) < .05
        self.assertGreater(independence_test(independent)['christoffersen_ind_p_value'], .05)
        quiet = independence_test([False]*100)
        self.assertEqual(quiet['christoffersen_ind_lr'], 0.0)
        self.assertTrue(np.isfinite(quiet['christoffersen_ind_p_value']))

    def test_load_returns_rejects_malformed_input(self):
        cases = {"unsorted dates": "Date,A\n2020-01-02,0.01\n2020-01-01,0.01\n",
                 "duplicate dates": "Date,A\n2020-01-01,0.01\n2020-01-01,0.02\n",
                 "missing value": "Date,A\n2020-01-01,0.01\n2020-01-02,\n",
                 "total loss": "Date,A\n2020-01-01,0.01\n2020-01-02,-1.0\n"}
        for description, content in cases.items():
            with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as handle:
                handle.write(content)
                path = handle.name
            try:
                with self.assertRaises(ValueError, msg=description):
                    load_returns(path)
            finally:
                os.unlink(path)
    
    def test_full_run_produces_every_output_and_finds_nothing_in_clean_noise(self):
        # run() writes every published file; nothing else tests it end to end.
        # Independent Gaussian returns must not trigger any rejection: if this
        # fails, a clustering result on real data cannot be trusted either.
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(rng.normal(0, .01, (320, 4)),
                             index=pd.bdate_range('2020-01-01', periods=320),
                             columns=['A', 'B', 'C', 'D'])
        with tempfile.TemporaryDirectory() as folder:
            data_path = pathlib.Path(folder)/'returns.csv'
            frame.to_csv(data_path, index_label='Date')
            output = pathlib.Path(folder)/'out'
            manifest = run(data_path, output)
            for name in ['rolling_forecasts.csv', 'coverage_summary.csv',
                         'risk_contributions.csv', 'run_manifest.json',
                         'stress_scenarios.json', 'RESULTS.md', 'risk_diagnostics.png']:
                self.assertTrue((output/name).exists(), name)
            self.assertEqual(manifest['return_observations'], 320)
            # Stress scenarios are asset-specific and must be skipped for other assets.
            self.assertEqual(json.loads((output/'stress_scenarios.json').read_text()), [])
            summary = pd.read_csv(output/'coverage_summary.csv')
            self.assertEqual(len(summary), 4)
            self.assertTrue((summary.conditional_coverage_p_value > .05).all())

if __name__=='__main__':
    unittest.main()
