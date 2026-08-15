import unittest

from simglucose.actuator.pump import InsulinPump
from simglucose.envs.simglucose_gym_env import (
    T1DSimEnv,
    T1DSimGymnaisumEnv,
)


class TestPumpUnits(unittest.TestCase):
    def setUp(self):
        self.pump = InsulinPump.withName("Insulet")

    def test_basal_is_quantized_and_clipped_in_units_per_hour(self):
        self.assertAlmostEqual(self.pump.basal(1.02 / 60), 1.0 / 60)
        self.assertAlmostEqual(self.pump.basal(1.03 / 60), 1.05 / 60)
        self.assertAlmostEqual(self.pump.basal(100 / 60), 30 / 60)
        self.assertEqual(self.pump.basal(-1), 0)

    def test_insulet_public_basal_limits_use_units_per_minute(self):
        self.assertEqual(self.pump.min_basal, 0.0)
        self.assertAlmostEqual(self.pump.max_basal, 0.5)

    def test_cozmo_basal_limit_uses_units_per_minute(self):
        pump = InsulinPump.withName("Cozmo")
        self.assertAlmostEqual(pump.basal(100 / 60), 35 / 60)
        self.assertAlmostEqual(pump.max_basal, 35 / 60)

    def test_gym_action_space_uses_units_per_minute(self):
        env = T1DSimEnv(patient_name="adolescent#001", seed=1)
        self.assertAlmostEqual(float(env.action_space.high[0]), 0.5)

    def test_gymnasium_action_space_uses_units_per_minute(self):
        env = T1DSimGymnaisumEnv(patient_name="adolescent#001", seed=1)
        self.assertAlmostEqual(float(env.action_space.high[0]), 0.5)


if __name__ == "__main__":
    unittest.main()
