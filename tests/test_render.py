from simglucose.simulation.rendering import Viewer
from datetime import datetime
import pandas as pd
import unittest
import logging
import os

logger = logging.getLogger(__name__)
TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'sim_results.csv')


class TestRendering(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(TESTDATA_FILENAME, index_col=0)
        self.df.index = pd.to_datetime(self.df.index)

    def test_rendering(self):
        start_time = datetime(2018, 1, 1, 0, 0, 0)
        viewer = Viewer(start_time, 'adolescent#001')
        for i in range(len(self.df)):
            df_tmp = self.df.iloc[0:(i + 1), :]
            viewer.render(df_tmp)
        viewer.close()


if __name__ == '__main__':
    unittest.main()
