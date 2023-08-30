from simglucose.analysis.report import risk_index_trace
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.rendering import Viewer
from datetime import datetime
import pandas as pd
import unittest
import logging
import os

logger = logging.getLogger(__name__)
TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'sim_results.csv')


class TestReport(unittest.TestCase):
    def setUp(self):
        self.df = pd.concat([pd.read_csv(TESTDATA_FILENAME, index_col=0)],  keys=['test'])

    def test_risk_index_trace(self):
        BG = self.df.unstack(level=0).BG
        sample_time = CGMSensor.withName("Dexcom").sample_time
        ri_per_hour, ri_mean, fig, axes = risk_index_trace(BG, sample_time=sample_time)

        LBGI = ri_per_hour.transpose().LBGI
        HBGI = ri_per_hour.transpose().HBGI
        RI = ri_per_hour.transpose()["Risk Index"]

        self.assertEqual(LBGI.size, 48)
        self.assertEqual(round(LBGI.iloc[-1].test, 3), 0.843)
        self.assertEqual(round(LBGI.iloc[0].test, 3), 0.0)

        self.assertEqual(HBGI.size, 48)
        self.assertEqual(round(HBGI.iloc[-1].test,3), 0.0)
        self.assertEqual(round(HBGI.iloc[0].test,3), 2.755)

        self.assertEqual(RI.size, 48)
        self.assertEqual(round(RI.iloc[-1].test,3), 0.843)
        self.assertEqual(round(RI.iloc[0].test,3), 2.755)


if __name__ == '__main__':
    unittest.main()
