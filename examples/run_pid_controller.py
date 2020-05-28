from simglucose.controller.pid_ctrller import PIDController
from simglucose.simulation.user_interface import simulate


pid_controller = PIDController(P=0.001, I=0.00001, D=0.001, target=140)
s = simulate(controller=pid_controller)
