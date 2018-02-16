import unittest
from unittest.mock import patch
from simglucose.simulation.user_interface import simulate
import shutil
import os

output_folder = os.path.join(os.path.dirname(__file__), 'results')


class testUI(unittest.TestCase):
    def setUp(self):
        pass

    @patch('builtins.input')
    def test_ui(self, mock_input):
        # animation, parallel, save_path, sim_time, scenario, scenario random
        # seed, start_time, patients, sensor, sensor seed, insulin pump,
        # controller
        mock_input.side_effect = ['n', 'y', output_folder, '24', '1', '2',
                                  '6', '1', '1', '1', '2', '1']
        s = simulate()
        self.assertEqual(s, 0)

    def tearDown(self):
        shutil.rmtree(output_folder)


if __name__ == '__main__':
    unittest.main()
