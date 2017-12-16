from simglucose.sensor.cgm import CGMSensor
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.patient.t1dpatient import Action
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

p = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
basal = p._params.u2ss * p._params.BW / 6000  # U/min
t = []
CHO = []
insulin = []
BG = []
CGM = []
while p.t < 1000:
    ins = basal
    carb = 0
    if p.t == 100:
        carb = 80
        ins = 80.0 / 6.0 + basal
        # if p.t == 150:
        #     ins = 80.0 / 12.0 + basal
    act = Action(insulin=ins, CHO=carb)
    t.append(p.t)
    CHO.append(act.CHO)
    insulin.append(act.insulin)
    BG.append(p.observation.Gsub)
    CGM.append(sensor.measure(p))
    p.step(act)
# print(CGM)
fig, ax = plt.subplots(3, sharex=True)
ax[0].plot(t, BG)
ax[0].plot(t, CGM, 'o', markersize=1)
ax[1].plot(t, CHO)
ax[2].plot(t, insulin)
plt.show()
