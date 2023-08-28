#!/usr/bin/env python

import sys
import numpy as np
import time
from casadi import *
from pyquaternion import Quaternion
import rospy

sys.path.insert(0, "/home/jerluennn/catkin_ws/src/icra2024-giraffe-efficient-design/continuum_robot_scripts")
sys.path.insert(0, "~/catkin_ws/src/icra2024-giraffe-efficient-design/tether_unit_scripts")

from generate_multiple_shooting_solver import Multiple_Shooting_Solver
from generate_robot_arm_model import Robot_Arm_Model
from generate_robot_arm_parameters import Robot_Arm_Params
from quasistatic_control_manager import Quasistatic_Control_Manager
from linear_mpc import Linear_MPC

from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray

class continuum_robot_inverse_kinematics:

    def __init__(self, manager): 

        self.manager = manager
        self.initialise_solver()
        self.angular_velocity = np.zeros(3)
        self.k_gain = 0.02
        self.k_gain_gripper = 0.5
        self.k_gain_tension_loss = 2
        self.initialise_pub_sub()   
        self.desired_tension_numpy = np.zeros(4)
        self.main_loop()
        
        

        rospy.spin()
        
    def callback_conversion(self, msg): 


        self.angular_velocity[0] = self.k_gain*msg.axes[3]
        self.angular_velocity[1] = -self.k_gain*msg.axes[4]
        self.desired_tension_numpy += self.k_gain_gripper*msg.buttons[3]
        self.desired_tension_numpy -= self.k_gain_gripper*msg.buttons[0]
        print(self.manager._current_states[0:7])

    def initialise_solver(self): 

        init_sol = np.zeros(19)
        init_sol[3] = 1
        init_sol[2] = -0.15
        init_sol[9] = -0

        self.manager.initialise_static_solver(init_sol)
        self.manager.set_tensions_static_MS_solver([0.0, 0.0, 0])
        self.manager.solve_static()

        self.manager.solve_Jacobians()

    def publish_tensions(self): 
        
        self.desired_tension_numpy[0:3] = self.k_gain_tension_loss * self.manager._current_tau * (1000/9.81)
        self.ref_tensions_before_curv_comp_msg.data = self.desired_tension_numpy
        self.ref_tensions_publisher.publish(self.ref_tensions_before_curv_comp_msg)
        self.simulated_pose_msg.data = self.manager._current_states[0:7]
        self.simulated_pose_publisher.publish(self.simulated_pose_msg)

    def initialise_pub_sub(self): 

        rospy.init_node('Continuum_Robot_Inverse_Kinematics', anonymous=True)
        rospy.Subscriber('/joy', Joy, self.callback_conversion)
        self.ref_tensions_publisher = rospy.Publisher('/ref_tensions', Float64MultiArray, queue_size=1)
        self.simulated_pose_publisher = rospy.Publisher('/simulated_pose', Float64MultiArray, queue_size=1)
        self.ref_tensions_before_curv_comp_msg = Float64MultiArray()
        self.simulated_pose_msg = Float64MultiArray()

    def main_loop(self):

        self.manager.apply_tension_differential(np.zeros(3))

        while not rospy.is_shutdown(): 

            tension_input = self.manager.solve_differential_inverse_kinematics(self.angular_velocity)
            self.manager.apply_tension_differential(tension_input)
            self.publish_tensions()
            time.sleep(1e-5)

scale = 2
tendon_radiuses_list = [[0.01250*scale, 0, 0], [-0.0051*scale, 0.0089*scale, 0], [-0.0051*scale, -0.0089*scale, 0]]
tendon_radiuses = SX(tendon_radiuses_list)
robot_arm_1 = Robot_Arm_Params(0.4, 0.05, -0.5, "1", 0.1)
robot_arm_1.from_hollow_rod(0.001, 0.0005, 70e9, 200e9)
robot_arm_1.set_mass_distribution(0.278)
robot_arm_1.set_gravity_vector('x')
C = np.diag([0.000, 0.000, 0.000])
Bbt = np.diag([1e-4, 1e-4, 1e-4])
Bse = Bbt
# Bse = np.zeros((3,3))
# Bbt = np.zeros((3,3))
robot_arm_1.set_damping_coefficient(C)
robot_arm_1.set_damping_factor(Bbt, Bse)
robot_arm_1.set_tendon_radiuses(tendon_radiuses_list)
robot_arm_model_1 = Robot_Arm_Model(robot_arm_1)

Q_w_p = 1000e3*np.eye(4)
Q_w_t = 1e-1*np.eye(3)
n_states = 4
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
init_sol[2] = 0
init_sol[9] = -0

quasi_sim_manager = Quasistatic_Control_Manager(robot_arm_model_1, diff_inv_solver)
continuum_robot_inverse_kinematics(quasi_sim_manager)