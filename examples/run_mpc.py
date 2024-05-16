import pandas as pd
from datetime import timedelta, datetime
from simglucose.controller.mpc_ctrller import PatientModel, MPCController
from simglucose.simulation.user_interface import simulate
from simglucose.simulation.scenario_gen import RandomScenario

RESULT_PATH = './results/sysid_results/adolescent#001.csv'
df_result = pd.read_csv(RESULT_PATH)
df_result = df_result.iloc[:-1, :]
patient_name = 'adolescent#001'
na = 3
nb = 5
nc = 5
X = df_result.loc[:, ['insulin', 'CHO']].values
y = df_result['CGM'].values
mdl = PatientModel(patient_name, na, nb, nc, X, y)

# print(mdl.params)
# print(mdl.alpha)
# print(mdl.beta)

N = 60  # horizon 5hr
Q = 1  #
QN = 10  # final cost
R = 1000  # input penality

ctrller = MPCController(patient_name, mdl, N, Q, QN, R, insulin_max=1)

SIM_RESULT_PATH = './results/mpc_results'
start_time = datetime.combine(datetime.now().date(),
                              datetime.min.time()) + timedelta(hours=7)
simulate(sim_time=timedelta(hours=24),
         scenario=RandomScenario(start_time=start_time, seed=1),
         patient_names=['adolescent#001'],
         cgm_name='Dexcom',
         cgm_seed=1,
         insulin_pump_name='Insulet',
         controller=ctrller,
         start_time=start_time,
         save_path=SIM_RESULT_PATH,
         animate=True,
         parallel=False)
