from datetime import timedelta, datetime
from simglucose.controller.sys_id_controller import SysIDController
from simglucose.simulation.user_interface import simulate
from simglucose.simulation.scenario import CustomScenario

RESULT_PATH = './results/sysid_results'

start_time = datetime.combine(datetime.now().date(), datetime.min.time())

insulin_time = [start_time + timedelta(hours=10)]
insulin_amount = [0.5]
sysid_controller = SysIDController(insulin_time, insulin_amount)

simulate(sim_time=timedelta(hours=24),
         scenario=CustomScenario(start_time=start_time,
                                 scenario=[(start_time + timedelta(hours=7),
                                            30)]),
         controller=sysid_controller,
         start_time=start_time,
         save_path=RESULT_PATH,
         animate=True,
         parallel=True)
