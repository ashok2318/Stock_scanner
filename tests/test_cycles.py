import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from cycle_analysis import analyze_cycles, repeated_zones
from pathlib import Path

class CycleTests(unittest.TestCase):
    def test_known_cycles_and_recovery(self):
        prices = pd.Series([100,110,100,90,100,110,100,90,100,110,100], index=pd.bdate_range('2026-01-01', periods=11))
        close, points, cycles = analyze_cycles(prices, 5, 1)
        self.assertEqual(list(points.kind), ['High','Low','High','Low','High'])
        self.assertEqual(len(cycles), 1)
        row = cycles.iloc[0]
        self.assertEqual(row['Cycle trading days'], 4)
        self.assertEqual(row['Rally trading days'], 2)
        self.assertEqual(row['Pullback trading days'], 2)
        self.assertEqual(row['Recovery trading days'], 2)
        self.assertAlmostEqual(row['Pullback %'], (90/110-1)*100)
        self.assertTrue((points.confirmed > points.date).all())
        self.assertEqual(len(repeated_zones(points)), 2)

    def test_flat_monotonic_missing_and_unrecovered(self):
        for prices in [[100]*20, list(range(100,120)), []]:
            _, points, cycles = analyze_cycles(pd.Series(prices, index=pd.bdate_range('2026-01-01', periods=len(prices)), dtype=float))
            self.assertTrue(points.empty)
            self.assertTrue(cycles.empty)
        prices = pd.Series([100,110,100,90,100,110,100,90,100], index=pd.bdate_range('2026-01-01', periods=9))
        _, _, cycles = analyze_cycles(prices, 5, 1)
        self.assertTrue(pd.isna(cycles.iloc[0]['Recovery trading days']))

    def test_page(self):
        class FakeTicker:
            def __init__(self, symbol): pass
            def history(self, **kwargs):
                x = 100+15*np.sin(np.arange(252)/8)
                return pd.DataFrame({'Close':x}, index=pd.bdate_range('2025-10-27', periods=252))
        with patch('yfinance.Ticker', FakeTicker):
            # Select the independent page before dashboard data calls.
            app = AppTest.from_string("import streamlit as st\nfrom cycle_analysis import render_cycle_page\nrender_cycle_page()")
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertGreater(len(app.dataframe), 1)
            app.radio[0].set_value('Calendar days').run()
            self.assertFalse(app.exception)
            app.slider[0].set_value(20.).run()
            self.assertFalse(app.exception)

if __name__ == '__main__': unittest.main()
