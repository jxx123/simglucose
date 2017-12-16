from simglucose.simulation.sim_env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.pump.insulin_pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import ScenarioGenerator
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter(
    '%(name)s: %(levelname)s: %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')
scenario = ScenarioGenerator(seed=1)
controller = BBController()

simEnv = T1DSimEnv(patient, sensor, pump, scenario)
action = controller.policy(patient.name, 0, patient.observation.Gsub)

plt.ion()
fig, axes = plt.subplots(4)
plt.show()
for _ in range(1440):
    observation = simEnv.step(action)
    simEnv.render(axes)

    CHO = simEnv.CHO_hist[-1]
    aciton = controller.policy(patient.name, CHO, simEnv.observation.CGM)
