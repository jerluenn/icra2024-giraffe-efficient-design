#!/usr/bin/env python

import roslib
import rospy
from geometry_msgs.msg import *
from std_msgs.msg import *

import time
import numpy as np
from pyquaternion import Quaternion

import sys
sys.path.insert(0, "/home/jerluennn/catkin_ws/src/icra2024-giraffe-efficient-design/tether_unit_scripts")
from atu_inverse_solver import atu_solver 

class atu_node: 

    def __init__(self, num_atu, pos_w_p, pos_centroid_d, robot_dict, friction_coefficient, initial_pose, initial_wrench, quat_drone_arm): 

        self.num_atu = num_atu
        self.robot_dict = robot_dict
        self.friction_coefficient = friction_coefficient
        # self.pos_w_centroid = np.array([-self.robot_dict['tether_length']*0.5, 0.0, self.robot_dict['tether_length']*0.8, 1, 0, 0, 0])
        self.pos_w_centroid = initial_pose
        self.pos_w_d = np.zeros((num_atu, 7))
        self.initconditions = np.zeros((num_atu, 14))
        self.ref = np.zeros((num_atu, 14))
        self.wrench_list = np.zeros((num_atu, 6))
        self.pos_centroid_d = pos_centroid_d
        self.median_i = 0
        self.median_pose = np.zeros((7, 7))
        self.pos_w_centroid_pose_feedback = np.zeros(7)
        self.quat = Quaternion([1, 0, 0, 0])
        self.wrench_msg = WrenchStamped()
        self.curvature_msg = Float64MultiArray()
        self.desired_tensions_msg = Float64MultiArray()
        self.curvature = np.zeros(num_atu) 
        self.median_k = 0 
        self.loadcell_measurements_filtered = np.zeros(num_atu)
        self.median_loadcell_measurements = np.zeros((7, num_atu))
        self.quat_w_mav = Quaternion(array=np.array([0.7071, 0, -0.7071, 0]))
        self.quat_drone_arm = Quaternion(array=quat_drone_arm)

        self.initialise_inverse_solvers(pos_w_p, initial_wrench)
        self.initialise_6dof_sensor(pos_centroid_d) 
        self.initialiseNode()

    def initialiseNode(self):

        rospy.init_node('atu_node')

        self.initialisePublishers()
        self.initialiseSubscribers()
        rospy.loginfo("Node starting!")
        self.mainLoop()
        rospy.spin()

    def initialise_inverse_solvers(self, pos_w_p, initial_wrench): 

        self.inverse_solvers_list = []
        self.integrators_list = []

        MAX_WRENCH = 10

        for i in range(self.num_atu):
             
            solver_obj = atu_solver(self.robot_dict)
            solver, integrator = solver_obj.createSolver() 
            self.inverse_solvers_list.append(solver)
            self.integrators_list.append(integrator)

            next_step_sol = np.zeros(14)
            next_step_sol[0:7] = pos_w_p[i]
            next_step_sol[7:13] = initial_wrench
            lbx = np.zeros(14)
            ubx = np.zeros(14)
            lbx[0:7] = pos_w_p[i]
            lbx[7:13] = -MAX_WRENCH, -MAX_WRENCH, -MAX_WRENCH, -MAX_WRENCH, -MAX_WRENCH, -MAX_WRENCH
            ubx[0:7] = pos_w_p[i]
            ubx[7:13] = MAX_WRENCH, MAX_WRENCH, MAX_WRENCH, MAX_WRENCH, MAX_WRENCH, MAX_WRENCH

            self.inverse_solvers_list[i].set(0, 'x', next_step_sol)
            self.inverse_solvers_list[i].constraints_set(0, 'lbx', lbx)
            self.inverse_solvers_list[i].constraints_set(0, 'ubx', ubx)

            for k in range(self.robot_dict['integration_steps']): 

                self.integrators_list[i].set('x', next_step_sol)
                self.integrators_list[i].solve()
                next_step_sol = self.integrators_list[i].get('x')
                self.inverse_solvers_list[i].set(k+1, 'x', next_step_sol) 

    def initialise_6dof_sensor(self, pos_centroid_d):

        NUM_ITERATIONS = 2000

        for i in range(self.num_atu):

            self.pos_w_d[i, 0:3] = self.pos_w_centroid[0:3] + pos_centroid_d[i, 0:3]
            self.pos_w_d[i, 3] = 1
            self.ref[i, 0:7] = self.pos_w_d[i, 0:7]
            self.inverse_solvers_list[i].set(self.robot_dict['integration_steps'], 'yref', self.ref[i, :])

            for k in range(NUM_ITERATIONS):

                self.inverse_solvers_list[i].solve()
                
                if self.inverse_solvers_list[i].get_cost() < 1e-8: 

                    break 


    def initialiseSubscribers(self):

        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.relative_pose_callback)
        rospy.Subscriber('/loadcell_measurements', Float64MultiArray, self.load_cell_measurements_callback)
        rospy.Subscriber('/ref_tensions_before_curv_comp', Float64MultiArray, self.update_desired_tensions_callback)

    def initialisePublishers(self):

        self.wrench_publisher = rospy.Publisher('/atu_wrench', WrenchStamped, queue_size=1)
        self.desired_tension_publisher = rospy.Publisher('/desired_tension_updated', Float64MultiArray, queue_size=1)
        self.curvature_publisher = rospy.Publisher('/curvature', Float64MultiArray, queue_size=1)

    def publish_data(self): 

        self.wrench_publisher.publish(self.wrench_msg)

    def update_desired_tensions_callback(self, solver_msg):

        self.desired_tensions_msg.data = solver_msg.data * (1/np.e**(-self.friction_coefficient*self.curvature))
        self.desired_tension_publisher.publish(self.desired_tensions_msg)

    def solve_structure_problem(self): 

        self.wrench_numpy = np.zeros(6)

        for i in range(self.num_atu):

            self.wrench_numpy += self.wrench_list[i] 
            self.wrench_numpy[3:6] += np.cross(self.quat.rotate(self.pos_centroid_d[i, 0:3]), self.wrench_list[i, 0:3])

        try:      

            ### Over here, we transform to body frame to compare with results from wrench sensor. 

            self.wrench_numpy[0:3] = self.quat.inverse.rotate(self.wrench_numpy[0:3])
            self.wrench_numpy[3:6] = self.quat.inverse.rotate(self.wrench_numpy[3:6])

            pass

        except:

            pass

        # R_sensor_body = np.array([[0, -1, 0],[-1, 0, 0],[0, 0, 1]])

        # self.wrench_numpy[0:3] = R_sensor_body@self.wrench_numpy[0:3]
        # self.wrench_numpy[3:6] = R_sensor_body@self.wrench_numpy[3:6]

        for i in range(self.num_atu): 

            self.wrench_numpy[2] += self.distal_tension[i]

        self.wrench_numpy[0:3] = self.quat_w_mav.inverse.rotate(self.wrench_numpy[0:3])
        self.wrench_numpy[3:6] = self.quat_w_mav.inverse.rotate(self.wrench_numpy[3:6])

        # self.wrench_numpy[0:3] = self.quat_w_drone.inverse.rotate(self.wrench_numpy[0:3])
        # self.wrench_numpy[3:6] = self.quat_w_drone.inverse.rotate(self.wrench_numpy[3:6])

        self.wrench_msg.wrench.force.x = -self.wrench_numpy[0]
        self.wrench_msg.wrench.force.y = -self.wrench_numpy[1]
        self.wrench_msg.wrench.force.z = -self.wrench_numpy[2]
        self.wrench_msg.wrench.torque.x = -self.wrench_numpy[3]
        self.wrench_msg.wrench.torque.y = -self.wrench_numpy[4]
        self.wrench_msg.wrench.torque.z = -self.wrench_numpy[5]

    def solve_distal_tension(self):
    
        self.distal_tension = self.loadcell_measurements_filtered*np.e**(-self.friction_coefficient*self.curvature)

    def update_curvatures(self): 

        for i in range(self.num_atu): 

            self.curvature[i] = self.inverse_solvers_list[i].get(self.robot_dict['integration_steps'], 'x')[-1]

        self.curvature_msg.data = self.curvature
        self.curvature_publisher.publish(self.curvature_msg)

    def update_solver(self):

        for i in range(self.num_atu): 

            self.quat[0] = self.pos_w_centroid[3]
            self.quat[1] = self.pos_w_centroid[4]
            self.quat[2] = self.pos_w_centroid[5]
            self.quat[3] = self.pos_w_centroid[6]
            self.ref[i, 0:3] = self.pos_w_centroid[0:3] + self.quat.rotate(self.pos_centroid_d[i, 0:3])
            self.ref[i, 3:7] = self.quat[0], self.quat[1], self.quat[2], self.quat[3]
            self.inverse_solvers_list[i].set(self.robot_dict['integration_steps'], 'yref', self.ref[i, :])
            self.inverse_solvers_list[i].solve()
            self.wrench_list[i, :] = self.inverse_solvers_list[i].get(self.robot_dict['integration_steps'], 'x')[7:13]

        self.update_curvatures()
        self.solve_distal_tension()
        self.solve_structure_problem()

    def medianFilterStep_pose_feedback(self): 

        self.median_pose[self.median_i%7, :] = self.pos_w_centroid_pose_feedback
        self.median_i += 1
        self.pos_w_centroid = np.median(self.median_pose[:, 0]), np.median(self.median_pose[:, 1]), np.median(self.median_pose[:, 2]),\
        np.median(self.median_pose[:, 3]), np.median(self.median_pose[:, 4]), np.median(self.median_pose[:, 5]), np.median(self.median_pose[:, 6])

    def medianFilterStep_loadcell(self): 

        self.median_loadcell_measurements[self.median_k%7, :] = self.load_cell_measurements
        self.median_k += 1 
        self.load_cell_measurements_filtered = np.median(self.median_loadcell_measurements, 0)

    def load_cell_measurements_callback(self, data): 

        self.load_cell_measurements = np.array([i for i in data.data]) 	
        self.load_cell_measurements *= (9.81/1000)

        self.medianFilterStep_loadcell()

    def relative_pose_callback(self, data):

        
        # mav frame FLU
        # world frame DLF

        # [[-1, 0, 0], [0, 1, 0], [0, 0, 1]] 
        # 0, 0, 2

        self.quat_mav_drone = Quaternion(array=np.array([np.abs(data.pose.orientation.w), np.sign(data.pose.orientation.w)*data.pose.orientation.x, 
                                                    np.sign(data.pose.orientation.w)*data.pose.orientation.y, np.sign(data.pose.orientation.w)*data.pose.orientation.z]))
        self.quat_w_drone = self.quat_w_mav * self.quat_mav_drone * self.quat_drone_arm

        p_w_mav = np.array([0, 0, 4])
        self.p_w_drone = p_w_mav + self.quat_w_mav.rotate(np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z]))

        self.pos_w_centroid_pose_feedback[0] = self.p_w_drone[0]
        self.pos_w_centroid_pose_feedback[1] = self.p_w_drone[1]
        self.pos_w_centroid_pose_feedback[2] = self.p_w_drone[2]
        self.pos_w_centroid_pose_feedback[3] = self.quat_w_drone[0]
        self.pos_w_centroid_pose_feedback[4] = self.quat_w_drone[1]
        self.pos_w_centroid_pose_feedback[5] = self.quat_w_drone[2]
        self.pos_w_centroid_pose_feedback[6] = self.quat_w_drone[3]  #quat_w_drone[3]

        time.sleep(0.0001)

        self.medianFilterStep_pose_feedback()

    def mainLoop(self): 

        while not rospy.is_shutdown(): 

            t0 = time.time()

            self.update_solver()
            self.publish_data()

            time.sleep(0.0001) 

            # print(f"Total time (s): {time.time() - t0}")

if __name__ == "__main__":

    robot_dict = {}
    robot_dict['type'] = 'hollow_rod'
    robot_dict['outer_radius'] = 0.002
    robot_dict['inner_radius'] = 0.0006
    robot_dict['elastic_modulus'] = 1.8e9
    robot_dict['mass_distribution'] = 0.035
    robot_dict['tether_length'] = 5.1
    robot_dict['shear_modulus'] = 0.75e9
    robot_dict['integration_steps'] = 100
        
    num_atu = 1 

    initial_pose = np.array([-1.5, 0, 4.0, 1, 0, 0, 0])
    quat_drone_arm = np.array([0.7071, 0, 0.7071, 0])

    # initial_pose = np.array([-1.5, 0, 4.0, 0.7071, 0, -0.7071, 0])
    # quat_drone_arm = np.array([1, 0, 0, 0])

    ground_positions = np.array([[0., 0., 0., 1, 0, 0, 0]])
    centroid_distal_positions = np.array([[0, 0, 0, 1, 0, 0, 0]])
    initial_wrench = np.array([robot_dict['tether_length'] * robot_dict['mass_distribution'] * 9.81, -7.21548500e-26, 
                               -3.62844316e-33, 4.22730307e-26, 0.278456470882048, -1.91589977e-24])
    atu_node(num_atu, ground_positions, centroid_distal_positions, robot_dict, 0.0569, initial_pose, initial_wrench, quat_drone_arm)