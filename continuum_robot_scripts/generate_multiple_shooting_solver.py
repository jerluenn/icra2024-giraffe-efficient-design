from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *

import numpy as np
from generate_robot_arm_model import Robot_Arm_Model

import time

from matplotlib import pyplot as plt

class Multiple_Shooting_Solver:

    def __init__(self, robot_arm_model): 
    
        self._robot_arm_model = robot_arm_model
        self._boundary_length = self._robot_arm_model.get_boundary_length()
        self._integration_steps = self._robot_arm_model.get_num_integration_steps()
        # self.create_static_solver()

    def create_static_solver_full_robot(self): 

        self.ocp = AcadosOcp()
        self.ocp.model = self._robot_arm_model.get_static_full_robot_arm_model()
        self.nx = self.ocp.model.x.size()[0]
        self.ocp.model.name = 'static_solver_full_robot'
        nu = self.ocp.model.u.size()[0]
        ny = self.nx + nu

        x = self.ocp.model.x
        u = self.ocp.model.u

        self.ocp.dims.N = self._integration_steps*2
        self.ocp.solver_options.qp_solver_iter_max = 1000
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = vertcat(self._robot_arm_model.get_point_force_first_arm() - x[7:13])
        self.ocp.cost.W = np.zeros((6, 6)) # set the cost during solving instead!
        self.ocp.cost.yref = np.zeros((6))
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr_e = vertcat(self._robot_arm_model.get_tendon_point_force() - x[7:13])
        # self.ocp.cost.W_e = np.identity(6)
        self.ocp.cost.W_e = np.zeros((6, 6))
        self.ocp.cost.yref_e = np.zeros((6))

        # self.ocp.solver_options.sim_method_num_steps = self._integration_steps*2
        self.ocp.solver_options.qp_solver_warm_start = 2

        self.ocp.solver_options.levenberg_marquardt = 0.001

        self.ocp.solver_options.regularize_method = 'CONVEXIFY'
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # 
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.tf = self._boundary_length*2

        self.wrench_lb = -50
        self.wrench_ub = 50
        self.pos_ub = 5
        self.eta_ub = 1.05
        self.tension_max = 50

        self.ocp.constraints.idxbx_0 = np.arange(self.nx)

        self.ocp.constraints.lbx_e = np.hstack((-np.ones(3)*self.pos_ub, -np.ones(4)*self.eta_ub, self.wrench_lb*np.ones(6), np.zeros(self.nx-14), self._boundary_length*2))

        self.ocp.constraints.ubx_e = np.hstack((np.ones(3)*self.pos_ub, np.ones(4)*self.eta_ub, self.wrench_ub*np.ones(6), np.ones(self.nx-14)*self.tension_max, self._boundary_length*2))   

        self.ocp.constraints.idxbx = np.arange(self.nx)

        self.ocp.constraints.lbx = np.hstack((-np.ones(3)*self.pos_ub, -np.ones(4)*self.eta_ub, self.wrench_lb*np.ones(6), np.zeros(self.nx-13)))

        self.ocp.constraints.ubx = np.hstack((np.ones(3)*self.pos_ub, np.ones(4)*self.eta_ub, self.wrench_ub*np.ones(6), np.ones(self.nx-13)*self.tension_max))

        self.ocp.constraints.idxbx_e = np.arange(self.nx)

        self.ocp.constraints.lbx_0 = np.hstack((np.array([0, 0, 0, 1, 0, 0, 0]), self.wrench_lb*np.ones(6), np.zeros(self.nx-14), 0))

        self.ocp.constraints.ubx_0 = np.hstack((np.array([0, 0, 0, 1, 0, 0, 0]), self.wrench_ub*np.ones(6), np.ones(self.nx-14)*self.tension_max, 0))

        self.ocp.constraints.ubu = np.array([0]) 
        self.ocp.constraints.lbu = np.array([0]) 
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.solver_options.nlp_solver_max_iter = 1

        self.ocp.code_export_directory = 'ocp_solver_full_robot' + self.ocp.model.name

        # AcadosOcpSolver.generate(self.ocp, json_file=f'{self.ocp.model.name}.json')
        # AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        
        # solver = AcadosOcpSolver.create_cython_solver(json_file=f'{self.ocp.model.name}.json')
        solver = AcadosOcpSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')
        integrator = AcadosSimSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')

        return solver, integrator        


    def create_static_solver_position_boundary(self): 

        self.ocp = AcadosOcp()
        # self.ocp.model = self._robot_arm_model.get_static_robot_arm_model()
        self.ocp.model = self._robot_arm_model.get_static_robot_arm_model_with_boundaries_model()
        self.nx = self.ocp.model.x.size()[0]
        self.ocp.model.name = 'static_solver_position_boundary'
        nu = self.ocp.model.u.size()[0]
        ny = self.nx + nu

        x = self.ocp.model.x
        u = self.ocp.model.u

        self.ocp.dims.N = self._integration_steps
        self.ocp.solver_options.qp_solver_iter_max = 400
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr_e = vertcat(self._robot_arm_model.get_point_force_position_boundary() - x[7:13])
        self.ocp.cost.W_e = np.identity(6)
        self.ocp.cost.yref_e = np.zeros((6))

        self.ocp.solver_options.sim_method_num_steps = self._integration_steps
        self.ocp.solver_options.qp_solver_warm_start = 2

        self.ocp.solver_options.levenberg_marquardt = 0.00001

        self.ocp.solver_options.regularize_method = 'CONVEXIFY'
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # 
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.tf = self._boundary_length

        self.wrench_lb = -50
        self.wrench_ub = 50
        self.pos_ub = 5
        self.eta_ub = 1.05
        self.tension_max = 50

        self.ocp.constraints.idxbx_0 = np.arange(self.nx)

        self.ocp.constraints.lbx_e = np.hstack((-np.ones(3)*self.pos_ub, -np.ones(4)*self.eta_ub, self.wrench_lb*np.ones(6), np.zeros(self.nx-13)))

        self.ocp.constraints.ubx_e = np.hstack((np.ones(3)*self.pos_ub, np.ones(4)*self.eta_ub, self.wrench_ub*np.ones(6), np.ones(self.nx-13)*self.tension_max))   

        self.ocp.constraints.idxbx = np.arange(self.nx)

        self.ocp.constraints.lbx = np.hstack((-np.ones(3)*self.pos_ub, -np.ones(4)*self.eta_ub, self.wrench_lb*np.ones(6), np.zeros(self.nx-13)))

        self.ocp.constraints.ubx = np.hstack((np.ones(3)*self.pos_ub, np.ones(4)*self.eta_ub, self.wrench_ub*np.ones(6), np.ones(self.nx-13)*self.tension_max))

        self.ocp.constraints.idxbx_e = np.arange(self.nx)

        self.ocp.constraints.lbx_0 = np.hstack((np.array([0, 0, 0, 1, 0, 0, 0]), self.wrench_lb*np.ones(6), np.zeros(self.nx-13)))

        self.ocp.constraints.ubx_0 = np.hstack((np.array([0, 0, 0, 1, 0, 0, 0]), self.wrench_ub*np.ones(6), np.ones(self.nx-13)*self.tension_max))

        self.ocp.constraints.ubu = np.array([0]) 
        self.ocp.constraints.lbu = np.array([0]) 
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.solver_options.nlp_solver_max_iter = 1

        self.ocp.code_export_directory = 'ocp_solver_position_boundary' + self.ocp.model.name

        # AcadosOcpSolver.generate(self.ocp, json_file=f'{self.ocp.model.name}.json')
        # AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        
        # solver = AcadosOcpSolver.create_cython_solver(json_file=f'{self.ocp.model.name}.json')
        solver = AcadosOcpSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')
        integrator = AcadosSimSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')

        return solver, integrator        

    def create_static_solver(self):

        self.ocp = AcadosOcp()
        self.ocp.model = self._robot_arm_model.get_static_robot_arm_model()
        self.nx = self.ocp.model.x.size()[0]
        nu = self.ocp.model.u.size()[0]
        ny = self.nx + nu

        x = self.ocp.model.x
        u = self.ocp.model.u

        self.ocp.dims.N = self._integration_steps
        self.ocp.solver_options.qp_solver_iter_max = 400
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # self.ocp.model.cost_y_expr_e = vertcat(self._robot_arm_model.get_tendon_point_force_opposite_direction() - x[7:13])
        self.ocp.model.cost_y_expr_e = vertcat(self._robot_arm_model.get_tendon_point_force() - x[7:13])
        self.ocp.cost.W_e = np.identity(6)
        self.ocp.cost.yref_e = np.zeros((6))

        self.ocp.solver_options.sim_method_num_steps = self._integration_steps
        self.ocp.solver_options.qp_solver_warm_start = 2

        self.ocp.solver_options.levenberg_marquardt = 0.000001

        self.ocp.solver_options.regularize_method = 'CONVEXIFY'
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # 
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.tf = self._boundary_length

        self.wrench_lb = -50
        self.wrench_ub = 50
        self.pos_ub = 5
        self.eta_ub = 1.05
        self.tension_max = 50

        self.ocp.constraints.idxbx_0 = np.arange(self.nx)

        self.ocp.constraints.lbx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self.wrench_lb*np.ones(6), np.zeros(self.nx - 13)))

        self.ocp.constraints.ubx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self.wrench_ub*np.ones(6), np.zeros(self.nx - 13)))        

        self.ocp.constraints.idxbx = np.arange(self.nx)

        self.ocp.constraints.lbx = np.hstack((-np.ones(3)*self.pos_ub, -np.ones(4)*self.eta_ub, self.wrench_lb*np.ones(6), np.zeros(self.nx-13)))

        self.ocp.constraints.ubx = np.hstack((np.ones(3)*self.pos_ub, np.ones(4)*self.eta_ub, self.wrench_ub*np.ones(6), np.ones(self.nx-13)*self.tension_max))

        self.ocp.constraints.ubu = np.array([0]) 
        self.ocp.constraints.lbu = np.array([0]) 
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.solver_options.nlp_solver_max_iter = 1

        self.ocp.code_export_directory = 'ocp_solver' + self.ocp.model.name

        # AcadosOcpSolver.generate(self.ocp, json_file=f'{self.ocp.model.name}.json')
        # AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        
        # solver = AcadosOcpSolver.create_cython_solver(json_file=f'{self.ocp.model.name}.json')
        solver = AcadosOcpSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')
        integrator = AcadosSimSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')

        return solver, integrator

    def create_dynamic_solver(self): 

        self.ocp = AcadosOcp()
        self.ocp.model = self._robot_arm_model.get_dynamic_robot_arm_model()
        self.nx = self.ocp.model.x.size()[0]
        nu = self.ocp.model.u.size()[0]
        ny = self.nx + nu

        x = self.ocp.model.x
        u = self.ocp.model.u

        self.ocp.dims.N = self._integration_steps
        self.ocp.solver_options.qp_solver_iter_max = 400
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr_e = vertcat(x[7:13] - self._robot_arm_model.get_tendon_point_force_dynamic())
        self.ocp.cost.W_e = np.identity(6)
        self.ocp.cost.yref_e = np.zeros((6))

        self.ocp.solver_options.sim_method_num_steps = self._integration_steps
        self.ocp.solver_options.qp_solver_warm_start = 2

        self.ocp.solver_options.levenberg_marquardt = 0.000001

        self.ocp.solver_options.regularize_method = 'CONVEXIFY'
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # 
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.tf = self._boundary_length
        self.ocp.parameter_values = np.zeros(12)

        self.wrench_lb = -50
        self.wrench_ub = 50
        self.pos_ub = 5
        self.eta_ub = 1.05
        self.tension_max = 50
        self.q_ub = 100
        self.om_ub = 100 

        self.ocp.constraints.idxbx_0 = np.arange(self.nx)

        # p, eta, n, m, q, om, tau

        self.ocp.constraints.lbx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self.wrench_lb*np.ones(6), -self.q_ub*np.ones(3), -self.om_ub*np.ones(3),np.zeros(self.nx - 19)))

        self.ocp.constraints.ubx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self.wrench_ub*np.ones(6), self.q_ub*np.ones(3), self.om_ub*np.ones(3), np.ones(self.nx-19)*self.tension_max))        

        self.ocp.constraints.idxbx = np.arange(self.nx)

        self.ocp.constraints.lbx = np.hstack((-np.ones(3)*self.pos_ub, -np.ones(4)*self.eta_ub, self.wrench_lb*np.ones(6), -self.q_ub*np.ones(3), -self.om_ub*np.ones(3), np.zeros(self.nx-19)))

        self.ocp.constraints.ubx = np.hstack((np.ones(3)*self.pos_ub, np.ones(4)*self.eta_ub, self.wrench_ub*np.ones(6), self.q_ub*np.ones(3), self.om_ub*np.ones(3), np.ones(self.nx-19)*self.tension_max))

        self.ocp.constraints.ubu = np.array([0]) 
        self.ocp.constraints.lbu = np.array([0]) 
        self.ocp.constraints.idxbu = np.array([0])

        self.ocp.code_export_directory = 'ocp_solver' + self.ocp.model.name

        self.ocp.solver_options.nlp_solver_max_iter = 1

        # AcadosOcpSolver.generate(self.ocp, json_file=f'{self.ocp.model.name}.json')
        # AcadosOcpSolver.build(self.ocp.code_export_directory, with_cython=True)
        
        # solver = AcadosOcpSolver.create_cython_solver(json_file=f'{self.ocp.model.name}.json')
        solver = AcadosOcpSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')
        integrator = AcadosSimSolver(self.ocp, json_file=f'{self.ocp.model.name}.json')

        return solver, integrator
            
    def set_tensions(self, solver, tension): 

        lbx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self.wrench_lb*np.ones(6), tension))
        ubx_0 = np.hstack((0, 0, 0, 1, 0, 0, 0, self.wrench_ub*np.ones(6), tension))        

        solver.constraints_set(0, 'lbx', lbx_0)
        solver.constraints_set(0, 'ubx', ubx_0)

