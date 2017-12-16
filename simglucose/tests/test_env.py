from simglucose.simulation.sim_env import T1DSimEnv, Observation
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.pump.insulin_pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import ScenarioGenerator
import matplotlib.pyplot as plt
import logging
import matplotlib.dates as mdates

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
obs = Observation(CHO=0, CGM=patient.observation.Gsub)
action = controller.policy(obs,
                           patient_name=patient.name,
                           sample_time=simEnv.sample_time)

plt.ion()
fig, axes = plt.subplots(4)
axes[0].tick_params(labelbottom='off')

axes[1].tick_params(labelbottom='off')

axes[2].tick_params(labelbottom='off')

axes[3].xaxis.set_minor_locator(mdates.AutoDateLocator())
axes[3].xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M\n'))
axes[3].xaxis.set_major_locator(mdates.DayLocator())
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('\n%b %d'))

plt.show(block=False)
for _ in range(100):
    obs, reward, done, info = simEnv.step(action)
    simEnv.render(axes)
    plt.pause(0.01)

    action = controller.policy(obs,
                               patient_name=patient.name,
                               sample_time=simEnv.sample_time)

print(simEnv.show_history().head())
