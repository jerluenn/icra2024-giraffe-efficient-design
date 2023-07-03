import sys
import numpy as np
import time
from casadi import *
from pyquaternion import Quaternion

sys.path.insert(0, "..")

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params
from quasistatic_control_manager import Quasistatic_Control_Manager
from linear_mpc import Linear_MPC

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### SETTING UP SOLVER ###

tendon_radiuses_list = [[0.0175, 0, 0], [-0.00875, 0.0151554, 0], [-0.00875, -0.0151554, 0]]
tendon_radiuses = SX(tendon_radiuses_list)
robot_arm_1 = Robot_Arm_Params(0.15, 0.05, -0.5, "1", 0.1)
robot_arm_1.from_solid_rod(0.0005, 100e9, 200e9, 8000)
robot_arm_1.set_gravity_vector('-z')
C = np.diag([0.000, 0.000, 0.000])
Bbt = np.diag([1e-4, 1e-4, 1e-4])
Bse = Bbt
# Bse = np.zeros((3,3))
# Bbt = np.zeros((3,3))
robot_arm_1.set_damping_coefficient(C)
robot_arm_1.set_damping_factor(Bbt, Bse)
robot_arm_1.set_tendon_radiuses(tendon_radiuses_list)
robot_arm_model_1 = Robot_Arm_Model(robot_arm_1)

Q_w_p = 1000e3*np.eye(2)
Q_w_t = 1e-1*np.eye(3)
n_states = 2
n_tendons = 3
name = 'single_controller'
R_mat = 1e-2*np.eye(3)
Tf = 0.01
q_max = 20 
q_dot_max = 5

diff_inv = Linear_MPC(Q_w_p, Q_w_t, n_states, n_tendons,q_dot_max, q_max, name, R_mat, Tf)
diff_inv_solver, _ = diff_inv.create_inverse_differential_kinematics()

initial_solution = np.zeros(19)
initial_solution[3] = 1

init_sol = np.zeros(19)
init_sol[3] = 1
init_sol[2] = -0.15
init_sol[9] = -0

# init_sol = np.array([ 2.48325142e-02, -3.11193014e-03,  1.46443802e-01, -2.30452729e-02,
#         9.66653197e-01, -4.44862301e-02, -2.51140405e-01,  2.41735197e+00,
#        -3.34391622e-01,  1.42101669e+00, -1.03478262e-02, -8.61231730e-02,
#         5.94504356e-03,  5.00000000e+00,  3.41327957e-12,  3.41326365e-12])

# integrator = robot_arm_model_1._create_static_integrator_with_boundaries()

quasi_sim_manager = Quasistatic_Control_Manager(robot_arm_model_1, diff_inv_solver)

quasi_sim_manager.initialise_static_solver(init_sol)
quasi_sim_manager.set_tensions_static_MS_solver([0.0, 0.0, 0])
quasi_sim_manager.solve_static()
# 

t0 = time.time()

quasi_sim_manager.solve_Jacobians()

N = 100
# quasi_sim_manager.set_time_step(1e-3)
# quasi_sim_manager.apply_tension_differential(np.zeros(3))

for i in range(N): 

    # tension_input = quasi_sim_manager.solve_differential_inverse_kinematics(np.array([-0.05, 0.1]))
    # quasi_sim_manager.apply_tension_differential(np.array(tension_input))
    quasi_sim_manager.apply_tension_differential(np.array([1.5, 0, 0]))
    # quasi_sim_manager.print_Jacobians_position_boundary()
    # quasi_sim_manager.save_step()


# print(quasi_sim_manager.get_simulation_data()[1][0:3, -1])    
# print(quasi_sim_manager.get_simulation_data()[1][13:, -1])
# print("----------------------------------------")
print(f"Time taken: {(time.time() - t0)/N}")

# quasi_sim_manager.print_Jacobians()
quasi_sim_manager.visualise_pb_arm()

# quasi_sim_manager.animate('test')

