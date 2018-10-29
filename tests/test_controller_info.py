import unittest
from simglucose.simulation.env import T1DSimEnv
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from datetime import datetime
from simglucose.controller.base import Action


class TestEnvInfo(unittest.TestCase):
    def test_env_info(self):
        now = datetime.now()
        start_time = datetime.combine(now.date(), datetime.min.time())

        # --------- Create Random Scenario --------------
        # Create a simulation environment
        patient = T1DPatient.withName('adolescent#001')
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')
        scenario = RandomScenario(start_time=start_time, seed=1)
        env = T1DSimEnv(patient, sensor, pump, scenario)

        obs, reward, done, info = env.reset()
        self.assertTrue('patient_state' in info.keys())

        action = Action(basal=0, bolus=0)
        obs, reward, done, info = env.step(action)
        self.assertTrue('patient_state' in info.keys())
