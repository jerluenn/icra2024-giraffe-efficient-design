import sys
sys.path.insert(0, '../../utils')
import os
import shutil
from datetime import datetime

from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *

import numpy as np

import time
import tetherunit
import rod_parameterbuilder
from matplotlib import pyplot as plt


class atu_solver:

    def __init__(self, robot_dict):

        self.build_tetherObject(robot_dict)
        self.boundary_length = robot_dict['tether_length']  # used for plotting
        self.integration_steps = robot_dict['integration_steps']
        self.createSolver()

    def build_tetherObject(self, robot_dict):

        builder = rod_parameterbuilder.Rod_Parameter_Builder()
        builder.createHollowRod(robot_dict)
        try:
            self.tetherObject = tetherunit.TetherUnit(builder, sys.argv[1])
        except:
            self.tetherObject = tetherunit.TetherUnit(builder)

    def createSolver(self):

        self.ocp = AcadosOcp()
        self.ocp.model = self.tetherObject.model
        nx = self.tetherObject.model.x.size()[0]
        nu = self.tetherObject.model.u.size()[0]
        ny = nx + nu

        self.ocp.dims.N = self.integration_steps
        # self.ocp.cost.cost_type_0 = 'LINEAR_LS'
        # self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.ocp.cost.W_e = np.identity(nx)
        self.ocp.cost.W = np.zeros((ny, ny))
        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx_e = np.zeros((nx, nx))
        self.ocp.cost.Vx_e[0:7, 0:7] = np.identity(7)
        self.ocp.cost.yref = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((nx))
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.solver_options.qp_solver_iter_max = 400
        # self.ocp.solver_options.sim_method_num_steps = self.integration_steps
        self.ocp.solver_options.qp_solver_warm_start = 2

        self.ocp.solver_options.levenberg_marquardt = 0.01

        # self.ocp.solver_options.levenberg_marquardt = 1.0

        # self.ocp.solver_options.levenberg_marquardt = 1.0
        self.ocp.solver_options.regularize_method = 'CONVEXIFY'

        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP'  # SQP_RTI, SQP
        self.ocp.solver_options.tf = self.boundary_length

        wrench_lb = -5
        wrench_ub = 5

        self.ocp.constraints.idxbx_0 = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

        self.ocp.constraints.lbx_0 = np.array([0, 0, 0, 1, 0, 0, 0,  # pose at start.
                                               wrench_lb, wrench_lb, wrench_lb, wrench_lb, wrench_lb, wrench_lb,
                                               0])  # tension, alpha, kappa, curvature

        self.ocp.constraints.ubx_0 = np.array([0, 0, 0, 1, 0, 0, 0,  # pose at start.
                                               wrench_ub, wrench_ub, wrench_ub, wrench_ub, wrench_ub, wrench_ub,
                                               0])

        self.ocp.constraints.idxbx = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

        self.ocp.constraints.lbx = np.array([-5, -5, -5, -1.05, -1.05, -1.05, -1.05,  # pose at start.
                                             wrench_lb, wrench_lb, wrench_lb, wrench_lb, wrench_lb, wrench_lb,
                                             0])

        self.ocp.constraints.ubx = np.array([5, 5, 5, 1.05, 1.05, 1.05, 1.05,  # pose at start.
                                             wrench_ub, wrench_ub, wrench_ub, wrench_ub, wrench_ub, wrench_ub,
                                             50])

        self.ocp.constraints.ubu = np.array([0])
        self.ocp.constraints.lbu = np.array([0])
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.solver_options.nlp_solver_max_iter = 1

        solver = AcadosOcpSolver(
            self.ocp, json_file=f'{self.ocp.model.name}.json')
        integrator = AcadosSimSolver(
            self.ocp, json_file=f'{self.ocp.model.name}.json')

        return solver, integrator


if __name__ == "__main__":

    robot_dict = {}
    robot_dict['type'] = 'hollow_rod'
    robot_dict['outer_radius'] = 0.002
    robot_dict['inner_radius'] = 0.0006
    robot_dict['elastic_modulus'] = 1.0e9
    robot_dict['mass_distribution'] = 0.035
    robot_dict['tether_length'] = 3.1
    robot_dict['shear_modulus'] = 0.75e9
    robot_dict['integration_steps'] = 50

    NUM_ITERATIONS = 1500

    solver_obj = atu_solver(robot_dict)
    solver, integrator = solver_obj.createSolver()
    yref = np.zeros((14, ))

    yref[0:7] = -robot_dict['tether_length'] * \
        0.8, 0.0, robot_dict['tether_length'] * 0.5, 1, 0, 0, 0

    solver.cost_set(robot_dict['integration_steps'], 'yref', yref)

    next_step_sol = np.array([0, 0, 0, 1, 0, 0, 0, robot_dict['tether_length'] * robot_dict['mass_distribution']
                              * 9.81, -7.21548500e-26, -3.62844316e-33, 4.22730307e-26, 0.1611493627, -1.91589977e-24, 0])
    # next_step_sol = np.array([0, 0, 0, 1, 0, 0, 0, 1.35202744e-01,  8.59444117e-11,  1.38997104e-01, -3.21851497e-11,  6.09179901e-03, -5.40886376e-12, 0])
    solver.set(0, 'x', next_step_sol)

    for i in range(robot_dict['integration_steps']):

        integrator.set('x', next_step_sol)
        integrator.solve()
        next_step_sol = integrator.get('x')
        print("next_step_sol: ", next_step_sol)
        solver.set(i + 1, 'x', next_step_sol)

    prev_sol = np.zeros(((robot_dict['integration_steps']+1)*NUM_ITERATIONS, 14))

    solver.cost_set(robot_dict['integration_steps'], 'yref', yref)

    # for i in range(NUM_ITERATIONS):

    #     solver.solve()


    # yref[0:7] = -1.5, 0.5, 2.0, 1, 0, 0, 0
    # solver.cost_set(robot_dict['integration_steps'], 'yref', yref)
    # print(solver.get(0, 'x'))
    # ax = plt.figure()
    # plt.show()

    start_time = time.time()

    for i in range(NUM_ITERATIONS):

        for k in range(robot_dict['integration_steps']+1):

            prev_sol[(robot_dict['integration_steps']+1)*i + k, :] = solver.get(k, 'x')

        solver.solve()



    print(solver.get_cost())
    print(solver.get(robot_dict['integration_steps'], "x"))

    print(f"Total time (s): {time.time() - start_time}")

    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    np.savetxt('states' + current_datetime_str + '.csv', prev_sol, delimiter=',')

    solver.solve()
    solver.print_statistics()

    print(solver.get(0, "x"))
    print(solver.get(robot_dict['integration_steps'], "x"))
    print(solver.get_cost())
    # print(solver.get_residuals())
