import sys

from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *
import numpy as np
import scipy
from scipy import linalg
import time

class Linear_MPC: 

    def __init__(self, Q_weight_pose, Q_weight_tensions, n_states, n_tendons, q_dot_max, q_max, name, R_weight, Tf): 

        self._Q_weight_pose = Q_weight_pose
        self._Q_weight_tensions = Q_weight_tensions
        self._n_states = n_states
        self._n_tendons = n_tendons
        self._name = name 
        self._R_weight = R_weight
        self._Tf = Tf
        self._qdot_max = q_dot_max
        self._q_max = q_max

    def create_inverse_differential_kinematics(self):

        model_name = 'controller' + self._name

        q_dot = SX.sym('q_dot', self._n_tendons)
        q = SX.sym('q', self._n_tendons)
        x = SX.sym('x', self._n_states)
        J = SX.sym('J', self._n_states*self._n_tendons, 1)
        x_dot = SX.sym('x_dot', self._n_states)

        states = vertcat(x, q)
        states_dot = vertcat(x_dot, q_dot)
        xdot = reshape(J, self._n_states, self._n_tendons)@q_dot
        f_expl_expr = vertcat(xdot, q_dot)
        f_impl_expr = f_expl_expr - states_dot

        model = AcadosModel()
        model.name = self._name
        model.x = states
        model.xdot = states_dot
        model.u = q_dot
        model.f_expl_expr = f_expl_expr
        model.f_impl_expr = f_impl_expr
        model.z = []
        model.p = J

        ocp = AcadosOcp()
        ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        ocp.dims.N = 1

        ocp.code_export_directory = model_name

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.cost.W = scipy.linalg.block_diag(self._Q_weight_pose, self._Q_weight_tensions, self._R_weight)
        ocp.cost.W_e = scipy.linalg.block_diag(10e2*self._Q_weight_pose, 0.1*self._Q_weight_tensions)

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)

        Vu = np.zeros((ny, nu))
        Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vu = Vu

        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros(ny_e, )

        ocp.constraints.idxbu = np.array([i for i in range(nu)])
        ocp.constraints.lbu = np.array([-self._qdot_max for i in range(nu)])
        ocp.constraints.ubu = np.array([self._qdot_max for i in range(nu)])
        ocp.constraints.x0 = np.zeros(nx)

        ocp.constraints.idxbx_e = np.array([i+self._n_states for i in range(nx - self._n_states)])
        ocp.constraints.lbx_e = np.array([0 for i in range(nx - self._n_states)])
        ocp.constraints.ubx_e = np.array([self._q_max for i in range(nx - self._n_states)])

        ocp.solver_options.levenberg_marquardt = 0.01
        # ocp.solver_options.regularize_method = 'CONVEXIFY'
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'  # FULL_CONDENSING_QPOASES
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI
        ocp.solver_options.qp_solver_iter_max = 200
        ocp.solver_options.nlp_solver_max_iter = 10
        ocp.parameter_values = np.zeros(self._n_states*self._n_tendons)
        # ocp.solver_options.qp_solver_cond_N = self._N

        # set prediction horizon
        ocp.solver_options.tf = self._Tf

        # AcadosOcpSolver.generate(ocp, json_file=f'{ocp.model.name}.json')
        # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        
        # solver = AcadosOcpSolver.create_cython_solver(json_file=f'{ocp.model.name}.json')
        solver = AcadosOcpSolver(ocp, json_file=f'{ocp.model.name}.json')
        integrator = AcadosSimSolver(ocp, json_file=f'{ocp.model.name}.json')

        return solver, integrator

