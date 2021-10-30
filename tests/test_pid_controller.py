import unittest
from simglucose.controller.pid_ctrller import PIDController
from simglucose.simulation.user_interface import simulate
from unittest.mock import patch
import shutil
import os
import pandas as pd

output_folder = os.path.join(os.path.dirname(__file__), 'results')


class TestPIDController(unittest.TestCase):
    def setUp(self):
        pass

    @patch('builtins.input')
    def test_pid_controller(self, mock_input):
        pid_controller = PIDController(P=0.001, I=0.00001, D=0.001)
        # animation, parallel, save_path, sim_time, scenario, scenario random
        # seed, start_time, patients, sensor, sensor seed, insulin pump,
        # controller
        mock_input.side_effect = [
            'y', 'n', output_folder, '24', '1', '2', '6', '5', '1', 'd', '1',
            '1', '2', '1'
        ]
        results = simulate(controller=pid_controller)
        self.assertIsInstance(results, pd.DataFrame)

    def tearDown(self):
        shutil.rmtree(output_folder)


if __name__ == '__main__':
    unittest.main()
