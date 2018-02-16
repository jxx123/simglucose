import unittest
from pandas.util.testing import assert_frame_equal
import pandas as pd
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from datetime import datetime
import os
import logging
import shutil

logger = logging.getLogger(__name__)

TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'sim_results.csv')
save_folder = os.path.join(os.path.dirname(__file__), 'results')


class TestSimEngine(unittest.TestCase):
    def test_batch_sim(self):
        # specify start_time as the beginning of today
        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())

        # --------- Create Random Scenario --------------
        # Create a simulation environment
        patient = T1DPatient.withName('adolescent#001')
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')
        scenario = RandomScenario(start_time=start_time, seed=1)
        env = T1DSimEnv(patient, sensor, pump, scenario)

        # Create a controller
        controller = BBController()

        # Put them together to create a simulation object
        s1 = SimObj(env, controller, timedelta(
            days=2), animate=True, path=save_folder)
        results1 = sim(s1)

        # --------- Create Custom Scenario --------------
        # Create a simulation environment
        patient = T1DPatient.withName('adolescent#001')
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')
        # custom scenario is a list of tuples (time, meal_size)
        scen = [(7, 45), (12, 70), (16, 15), (18, 80), (23, 10)]
        scenario = CustomScenario(start_time=start_time, scenario=scen)
        env = T1DSimEnv(patient, sensor, pump, scenario)

        # Create a controller
        controller = BBController()

        # Put them together to create a simulation object
        s2 = SimObj(env, controller, timedelta(
            days=2), animate=False, path=save_folder)
        results2 = sim(s2)

        # --------- batch simulation --------------
        s1.reset()
        s2.reset()
        s1.animate = False
        s = [s1, s2]
        results_para = batch_sim(s, parallel=True)

        s1.reset()
        s2.reset()
        s = [s1, s2]
        results_serial = batch_sim(s, parallel=False)

        assert_frame_equal(results_para[0], results1)
        assert_frame_equal(results_para[1], results2)
        for r1, r2 in zip(results_para, results_serial):
            assert_frame_equal(r1, r2)

    def test_results_consistency(self):
        # Test data
        results_exp = pd.read_csv(TESTDATA_FILENAME, index_col=0)
        results_exp.index = pd.to_datetime(results_exp.index)

        # specify start_time as the beginning of today
        start_time = datetime(2018, 1, 1, 0, 0, 0)

        # --------- Create Random Scenario --------------
        # Create a simulation environment
        patient = T1DPatient.withName('adolescent#001')
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')
        scenario = RandomScenario(start_time=start_time, seed=1)
        env = T1DSimEnv(patient, sensor, pump, scenario)

        # Create a controller
        controller = BBController()

        # Put them together to create a simulation object
        s = SimObj(env, controller, timedelta(
            days=2), animate=False, path=save_folder)
        results = sim(s)
        assert_frame_equal(results, results_exp)

    def tearDown(self):
        shutil.rmtree(os.path.join(os.path.dirname(__file__), 'results'))


if __name__ == '__main__':
    unittest.main()
