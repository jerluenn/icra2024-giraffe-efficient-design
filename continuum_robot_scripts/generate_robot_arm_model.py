import sys
import os
import shutil

from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *

import numpy as np
from generate_robot_arm_parameters import Robot_Arm_Params

class Robot_Arm_Model: 

    def __init__(self, robot_arm_params): 

        self._dir_name = 'c_generated_code_' 
        self._robot_arm_params_obj = robot_arm_params
        self._integration_steps = 20
        self._integration_stages = 4
        self._build_robot_model()


    def _build_robot_model(self): 

        self._mass_body = self._robot_arm_params_obj.get_mass_body()
        self._mass_distribution = self._robot_arm_params_obj.get_mass_distribution()
        self._Kse = self._robot_arm_params_obj.get_Kse()
        self._Kbt = self._robot_arm_params_obj.get_Kbt()
        self._boundary_length = self._robot_arm_params_obj.get_arm_length()
        self._id = self._robot_arm_params_obj.get_id()
        self._tendon_radiuses = self._robot_arm_params_obj.get_tendon_radiuses()
        self._tendon_radiuses_numpy = self._robot_arm_params_obj._tendon_radiuses_numpy
        self._C = self._robot_arm_params_obj.get_C()
        self._Bbt = self._robot_arm_params_obj.get_Bbt()
        self._Bse = self._robot_arm_params_obj.get_Bse()
        self._initialise_states()
        self._create_static_integrator()
        # self._create_dynamic_integrator()
        self._create_static_full_robot_integrator()

    def _initialise_states(self):

        # Initialise all ODE states.

        self._p = SX.sym('p', 3)
        self._eta = SX.sym('self._eta', 4) 
        self._n = SX.sym('n', 3)
        self._m = SX.sym('m', 3)
        self._tau = SX.sym('tau', 3)
        self._tau1 = SX.sym('tau1', 3)
        self._q = SX.sym('q', 3)
        self._om = SX.sym('om', 3)
        self._b = SX.sym('b', 6)
        self._lengths = SX.sym('lengths', 3)

        self._p_d = SX.sym('p_dot', 3)
        self._eta_d = SX.sym('eta_dot', 4)
        self._n_d = SX.sym('n_dot', 3)
        self._m_d = SX.sym('m_dot', 3)
        self._tau_d = SX.sym('tau_dot', 3)
        self._tau1_d = SX.sym('tau1_dot', 3)
        self._q_d = SX.sym('q_dot', 3)
        self._om_d = SX.sym('om_dot', 3)
        self._alpha_d = SX.sym('alpha_dot', 1)
        self._b_d = SX.sym('boundary_dot', 6)
        self._lengths_d = SX.sym('lengths_dot', 3)

        # Initialise constants

        self._g = SX([9.81, 0, 0])

        if self._robot_arm_params_obj._g_direction == 'x': 

            self._g = SX([9.81, 0, 0])

        elif self._robot_arm_params_obj._g_direction == 'y': 

            self._g = SX([0, 9.81, 0])

        elif self._robot_arm_params_obj._g_direction == 'z': 

            self._g = SX([0, 0, 9.81])

        elif self._robot_arm_params_obj._g_direction == '-z': 

            self._g = SX([0, 0, -9.81])

        else: 

            pass 

        self._f_ext = self._mass_distribution * self._g
        self._Kappa = SX.sym('Kappa', 1)
        self._c0 = (1.5 + self._robot_arm_params_obj.get_alpha())/(self._robot_arm_params_obj.get_time_step()*(1 + self._robot_arm_params_obj.get_alpha()))
        self._c1 = -2/self._robot_arm_params_obj.get_time_step()
        self._c2 = (0.5 + self._robot_arm_params_obj.get_alpha())/(self._robot_arm_params_obj.get_time_step()*(1+self._robot_arm_params_obj.get_alpha()))
        self._d1 = self._robot_arm_params_obj.get_alpha()/(1+self._robot_arm_params_obj.get_alpha())
        self._growth_rate = 1.0

        # Intermediate states

       # Setting R 

        self._R = SX(3,3)
        self._R[0,0] = 2*(self._eta[0]**2 + self._eta[1]**2) - 1
        self._R[0,1] = 2*(self._eta[1]*self._eta[2] - self._eta[0]*self._eta[3])
        self._R[0,2] = 2*(self._eta[1]*self._eta[3] + self._eta[0]*self._eta[2])
        self._R[1,0] = 2*(self._eta[1]*self._eta[2] + self._eta[0]*self._eta[3])
        self._R[1,1] = 2*(self._eta[0]**2 + self._eta[2]**2) - 1
        self._R[1,2] = 2*(self._eta[2]*self._eta[3] - self._eta[0]*self._eta[1])
        self._R[2,0] = 2*(self._eta[1]*self._eta[3] - self._eta[0]*self._eta[2])
        self._R[2,1] = 2*(self._eta[2]*self._eta[3] + self._eta[0]*self._eta[1])
        self._R[2,2] = 2*(self._eta[0]**2 + self._eta[3]**2) - 1

        self._v_history = SX.sym('v_hist', 3)
        self._u_history = SX.sym('u_hist', 3)
        self._q_history = SX.sym('q_hist', 3)
        self._om_history = SX.sym('om_hist', 3)

        self._s = SX.sym('s', 1)

        steepness = 1000
        shift1 = 150

        self._n1 = SX.sym('n1', 3)
        self._m1 = SX.sym('m1', 3)
        self._n1_d = SX.sym('n1_d', 3)
        self._m1_d = SX.sym('m1_d', 3)

        self._sigmoid1 = 1 - (1/(1 + exp(-steepness*self._s + shift1)))
        self._sigmoid2 = (1/(1 + exp(-steepness*self._s + shift1)))

        self._u = inv(self._Kbt)@transpose(reshape(self._R, 3, 3))@self._m
        self._v = inv(self._Kse)@transpose(reshape(self._R, 3, 3))@self._n + SX([0, 0, 1])
        self._u_dyn = inv(self._robot_arm_params_obj.get_Kbt() + self._c0*self._robot_arm_params_obj.get_Bbt())@(transpose(reshape(self._R, 3, 3))@self._m - self._robot_arm_params_obj.get_Bbt()@self._u_history)
        self._v_dyn = inv(self._robot_arm_params_obj.get_Kse() + self._c0*self._robot_arm_params_obj.get_Bse())@(transpose(reshape(self._R, 3, 3))@self._n + self._robot_arm_params_obj.get_Kse()@SX([0, 0, 1]) - self._robot_arm_params_obj.get_Bse()@self._v_history)
        self._k = 0.01

        self._u_fr = self._sigmoid1*(inv(self._Kbt)@transpose(reshape(self._R, 3, 3))@self._m) + self._sigmoid2*(inv(self._Kbt)@transpose(reshape(self._R, 3, 3))@self._m1)
        self._v_fr = self._sigmoid1*(inv(self._Kse)@transpose(reshape(self._R, 3, 3))@self._n + SX([0, 0, 1])) + self._sigmoid2*(inv(self._Kse)@transpose(reshape(self._R, 3, 3))@self._n1 + SX([0, 0, 1]))

        self._v_t = self._c0*self._v_dyn + self._v_history
        self._u_t = self._c0*self._u_dyn + self._u_history

        # self._v_t = self._v_history
        # self._u_t = self._u_history
        self._q_t = self._c0*self._q + self._q_history
        self._om_t = self._c0*self._om + self._om_history


    def _create_static_integrator(self):

        model_name = self._dir_name + 'static_robot_arm' + self._id 

        c = self._k*(1-transpose(self._eta)@self._eta)

        u = SX.sym('u')
        lengths_dot = SX([])

        p_dot = reshape(self._R, 3, 3) @ self._v
        eta_dot = vertcat(
            0.5*(-self._u[0]*self._eta[1] - self._u[1]*self._eta[2] - self._u[2]*self._eta[3]),
            0.5*(self._u[0]*self._eta[0] + self._u[2]*self._eta[2] - self._u[1]*self._eta[3]),
            0.5*(self._u[1]*self._eta[0] - self._u[2]*self._eta[1] + self._u[0]*self._eta[3]),
            0.5*(self._u[2]*self._eta[0] + self._u[1]*self._eta[1] - self._u[0]*self._eta[2])
        ) + c * self._eta 
        n_dot = - (self._f_ext) - self.get_external_distributed_forces()
        # n_dot = - (self._f_ext) - self.get_external_distributed_forces_opposite_direction()
        m_dot = - cross(p_dot, self._n) 
        tau_dot = SX.zeros(self._tau.shape[0])

        for i in range(self._tau.size()[0]):

            # Currently accounts only for parallel tendons.
            lengths_dot = vertcat(lengths_dot, norm_2(
                p_dot + self._R@skew(self._u)@transpose(self._tendon_radiuses[i, :])))

        x = vertcat(self._p, self._eta, self._n, self._m, self._tau, self._lengths)
        xdot = vertcat(p_dot, eta_dot,
                       n_dot, m_dot, tau_dot, lengths_dot)
        x_dot_impl = vertcat(self._p_d, self._eta_d, self._n_d, self._m_d, self._tau_d, self._lengths_d)
        
        self._static_model = AcadosModel()
        self._static_model.name = model_name
        self._static_model.x = x 
        self._static_model.f_expl_expr = xdot 
        self._static_model.f_impl_expr = xdot - x_dot_impl
        self._static_model.u = u
        self._static_model.z = SX([])
        self._static_model.xdot = x_dot_impl

        

        sim = AcadosSim()
        sim.model = self._static_model 

        Sf = self._boundary_length

        sim.code_export_directory = model_name
        sim.solver_options.T = Sf
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = self._integration_stages
        sim.solver_options.num_steps = self._integration_steps
        # sim.solver_options.sens_forw = False

        acados_integrator = AcadosSimSolver(sim)

        return acados_integrator

    def _create_dynamic_integrator(self): 

        """TO DO:"""

        model_name = self._dir_name + 'dynamic_robot_arm' + self._id 

        c = self._k*(1-transpose(self._eta)@self._eta)

        u = SX.sym('u')

        p_dot = reshape(self._R, 3, 3) @ self._v_dyn
        eta_dot = vertcat(
            0.5*(-self._u_dyn[0]*self._eta[1] - self._u_dyn[1]*self._eta[2] - self._u_dyn[2]*self._eta[3]),
            0.5*(self._u_dyn[0]*self._eta[0] + self._u_dyn[2]*self._eta[2] - self._u_dyn[1]*self._eta[3]),
            0.5*(self._u_dyn[1]*self._eta[0] - self._u_dyn[2]*self._eta[1] + self._u_dyn[0]*self._eta[3]),
            0.5*(self._u_dyn[2]*self._eta[0] + self._u_dyn[1]*self._eta[1] - self._u_dyn[0]*self._eta[2])
        ) + c * self._eta 
        # n_dot = - (self._f_ext) - self.get_external_distributed_forces()
        # m_dot = - cross(p_dot, self._n) 
        n_dot = reshape(self._R, 3, 3) @ (self._robot_arm_params_obj.get_mass_distribution()*(skew(self._om)@self._q + self._q_t)) - self._robot_arm_params_obj.get_mass_distribution()*self._g + self._robot_arm_params_obj.get_C()@(self._q*self._q*(1/(1 + SX.exp(-self._growth_rate*self._q)))) - self.get_external_distributed_forces_dyn()
        m_dot = self._robot_arm_params_obj.get_rho() * reshape(self._R, 3, 3) @ (skew(self._om) @ self._robot_arm_params_obj.get_J() @ self._om + self._robot_arm_params_obj.get_J()@self._om_t) - skew(p_dot)@self._n

        q_dot = self._v_t - skew(self._u_dyn)@self._q + skew(self._om)@self._v_dyn
        om_dot = self._u_t - skew(self._u_dyn)@self._om

        # q_dot = self._v_t
        # om_dot = self._u_t

        tau_dot = SX.zeros(self._tau.shape[0])

        x = vertcat(self._p, self._eta, self._n, self._m, self._q, self._om, self._tau)
        xdot = vertcat(p_dot, eta_dot,
                       n_dot, m_dot, q_dot, om_dot, tau_dot)
        x_dot_impl = vertcat(self._p_d, self._eta_d, self._n_d, self._m_d, self._q_d, self._om_d,self._tau_d)

        self._dynamic_model = AcadosModel()
        self._dynamic_model.name = model_name
        self._dynamic_model.x = x 
        self._dynamic_model.f_expl_expr = xdot 
        self._dynamic_model.f_impl_expr = xdot - x_dot_impl
        self._dynamic_model.u = u
        self._dynamic_model.z = SX([])
        self._dynamic_model.xdot = x_dot_impl

        parameters_to_update = vertcat(self._v_history, self._u_history, self._q_history, self._om_history)

        self._dynamic_model.p = parameters_to_update

        sim = AcadosSim()
        sim.model = self._dynamic_model 

        Sf = self._boundary_length

        sim.code_export_directory = model_name
        sim.solver_options.T = Sf
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = self._integration_stages
        sim.solver_options.num_steps = self._integration_steps
        # sim.solver_options.sens_forw = False

        acados_integrator = AcadosSimSolver(sim)

        return acados_integrator

    def _create_static_integrator_with_boundaries(self): 

        """TO DO!"""

        model_name = self._dir_name + 'static_robot_arm_with_boundaries' + self._id 

        c = self._k*(1-transpose(self._eta)@self._eta)

        u = SX.sym('u')

        p_dot = reshape(self._R, 3, 3) @ self._v
        eta_dot = vertcat(
            0.5*(-self._u[0]*self._eta[1] - self._u[1]*self._eta[2] - self._u[2]*self._eta[3]),
            0.5*(self._u[0]*self._eta[0] + self._u[2]*self._eta[2] - self._u[1]*self._eta[3]),
            0.5*(self._u[1]*self._eta[0] - self._u[2]*self._eta[1] + self._u[0]*self._eta[3]),
            0.5*(self._u[2]*self._eta[0] + self._u[1]*self._eta[1] - self._u[0]*self._eta[2])
        ) + c * self._eta 
        n_dot = - (self._f_ext) - self.get_external_distributed_forces()
        # n_dot = - (self._f_ext) - self.get_external_distributed_forces_opposite_direction()
        m_dot = - cross(p_dot, self._n) 
        tau_dot = SX.zeros(self._tau.shape[0])

        b_pendulum = self.get_point_force_position_boundary() - vertcat(self._n, self._m)
        b = self.get_tendon_point_force() - vertcat(self._n, self._m)
        
        # b = self.get_tendon_point_force_opposite_direction() - vertcat(self._n, self._m)
        lengths_dot = SX([])

        self.db_pend_dy = jacobian(b_pendulum, vertcat(self._p, self._eta, self._n, self._m, self._tau))
        self.f_db_pend_dy = Function('f_db_pend_dy', [vertcat(self._p, self._eta, self._n, self._m, self._tau)], [self.db_pend_dy])
        self._f_b_pend = Function('f_b_pend', [vertcat(self._eta, self._n, self._m, self._tau)], [b_pendulum]) 

        self.db_dy = jacobian(b, vertcat(self._eta, self._n, self._m, self._tau))
        self.f_db_dy = Function('f_db_dy', [vertcat(self._eta, self._n, self._m, self._tau)], [self.db_dy])
        self._f_b = Function('f_b', [vertcat(self._eta, self._n, self._m, self._tau)], [b]) 

        for i in range(self._tau.size()[0]):

            # Currently accounts only for parallel tendons.
            lengths_dot = vertcat(lengths_dot, norm_2(
                p_dot + self._R@skew(self._u)@transpose(self._tendon_radiuses[i, :])))

        x = vertcat(self._p, self._eta, self._n, self._m, self._tau, self._lengths)
        xdot = vertcat(p_dot, eta_dot,
                       n_dot, m_dot, tau_dot, lengths_dot)

        self._static_model_with_boundaries_ = AcadosModel()
        self._static_model_with_boundaries_.name = model_name
        self._static_model_with_boundaries_.x = x 
        self._static_model_with_boundaries_.f_expl_expr = xdot 
        self._static_model_with_boundaries_.u = u
        self._static_model_with_boundaries_.z = SX([])

        sim = AcadosSim()
        sim.model = self._static_model_with_boundaries_ 

        Sf = self._boundary_length

        sim.code_export_directory = model_name
        sim.solver_options.T = Sf
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = self._integration_stages
        sim.solver_options.num_steps = self._integration_steps

        acados_integrator = AcadosSimSolver(sim)

        return acados_integrator

    def _create_static_full_robot_integrator(self): 
        
        pass

    # def get_external_distributed_forces_opposite_direction_full_robot(self): 


    def get_external_distributed_forces_second_arm(self): 

        p_dot = reshape(self._R, 3, 3) @ self._v

        p_dotdot = reshape(self._R, 3, 3) @ skew(self._u_fr) @ self._v

        f_t = SX.zeros(3)

        for i in range(self._tau.shape[0]):

            f_t -= (self._tau1[i]) * (skew(p_dot)@skew(p_dot))@p_dotdot / (norm_2(p_dot)**3)

        return f_t 


    def get_external_distributed_forces(self):

        """TO DO: Documentation"""

        p_dot = reshape(self._R, 3, 3) @ self._v

        p_dotdot = reshape(self._R, 3, 3) @ skew(self._u) @ self._v

        f_t = SX.zeros(3)

        for i in range(self._tau.shape[0]):

            f_t -= (self._tau[i]) * (skew(p_dot)@skew(p_dot))@p_dotdot / (norm_2(p_dot)**3)

        return f_t 

    def get_external_distributed_forces_opposite_direction(self):

        """TO DO: Documentation"""

        p_dot = - reshape(self._R, 3, 3) @ self._v

        p_dotdot = - reshape(self._R, 3, 3) @ skew(self._u) @ self._v

        f_t = SX.zeros(3)

        for i in range(self._tau.shape[0]):

            f_t -= (self._tau[i]) * (skew(p_dot)@skew(p_dot))@p_dotdot / (norm_2(p_dot)**3)

        return f_t 

    def get_tendon_point_force(self):

        # b = self.get_tendon_point_force() + vertcat(self._n, self._m)
        # b = self.get_tendon_point_force_opposite_direction() + vertcat(self._n, self._m)

        W_t = SX.zeros(6)

        p_dot = reshape(self._R, 3, 3) @ self._v

        for i in range(self._tau.shape[0]): 

            W_t -= vertcat(self._tau[i]*(p_dot/norm_2(p_dot)), self._tau[i]*skew(reshape(self._R, 3, 3)@transpose(self._tendon_radiuses[i, :]))@(p_dot/norm_2(p_dot)))

        return W_t
    
    def get_point_force_position_boundary(self):

        W_t = SX.zeros(6)

        # p_dot = reshape(self._R, 3, 3) @ np.linalg.inv(np.array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]])) @ self._v
        p_dot = reshape(self._R, 3, 3) @ self._v

        for i in range(self._tau.shape[0]): 

            W_t -= vertcat(self._tau[i]*(p_dot/norm_2(p_dot)), self._tau[i]*skew(reshape(self._R, 3, 3)@transpose(self._tendon_radiuses[i, :]))@(p_dot/norm_2(p_dot)))
            # W_t -= vertcat(self._tau1[i]*(p_dot/norm_2(p_dot)), self._tau1[i]*skew(reshape(self._R, 3, 3)@transpose(self._tendon_radiuses[i, :]))@(p_dot/norm_2(p_dot)))

        W_t += vertcat(self._mass_body*self._g, 0, 0, 0)

        return W_t

    def get_point_force_position_boundary1(self):

        W_t = SX.zeros(6)

        # p_dot = reshape(self._R, 3, 3) @ np.linalg.inv(np.array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]])) @ self._v
        p_dot = reshape(self._R, 3, 3) @ self._v

        for i in range(self._tau.shape[0]): 

            W_t -= vertcat(self._tau[i]*(p_dot/norm_2(p_dot)), self._tau[i]*skew(reshape(self._R, 3, 3)@transpose(self._tendon_radiuses[i, :]))@(p_dot/norm_2(p_dot)))
            W_t += vertcat(self._tau1[i]*(p_dot/norm_2(p_dot)), self._tau1[i]*skew(reshape(self._R, 3, 3)@transpose(self._tendon_radiuses[i, :]))@(p_dot/norm_2(p_dot)))

        W_t += vertcat(self._mass_body*self._g, 0, 0, 0)

        return W_t



    def get_point_force_first_arm(self):

        W_t = SX.zeros(6)

        # p_dot = reshape(self._R, 3, 3) @ np.linalg.inv(np.array([[-1, 0, 0],[0, 1, 0],[0, 0, -1]])) @ self._v
        p_dot = -reshape(self._R, 3, 3) @ self._v
        p_dot1 = reshape(self._R, 3, 3) @ self._v

        for i in range(self._tau.shape[0]): 

            W_t -= vertcat(self._tau[i]*(p_dot/norm_2(p_dot)), -self._tau[i]*skew(reshape(self._R, 3, 3)@transpose(self._tendon_radiuses[i, :]))@(p_dot/norm_2(p_dot)))
            W_t -= vertcat(self._tau1[i]*(p_dot1/norm_2(p_dot1)), self._tau1[i]*skew(reshape(self._R, 3, 3)@transpose(self._tendon_radiuses[i, :]))@(p_dot1/norm_2(p_dot1)))

        W_t += vertcat(self._mass_body*self._g, 0, 0, 0)

        return W_t

    def get_tendon_point_force_dynamic(self):

        W_t = SX.zeros(6)

        p_dot = reshape(self._R, 3, 3) @ self._v_dyn

        for i in range(self._tau.shape[0]): 

            W_t -= vertcat(self._tau[i]*(p_dot/norm_2(p_dot)), self._tau[i]*skew(reshape(self._R, 3, 3)@transpose(self._tendon_radiuses[i, :]))@(p_dot/norm_2(p_dot)))

        return W_t

    def get_external_distributed_forces_dyn(self):

        """TO DO: Documentation"""

        p_dot = reshape(self._R, 3, 3) @ self._v_dyn

        p_dotdot = reshape(self._R, 3, 3) @ skew(self._u_dyn) @ self._v_dyn

        f_t = SX.zeros(3)

        for i in range(self._tau.shape[0]):

            f_t -= (self._tau[i]) * (skew(p_dot)@skew(p_dot))@p_dotdot / (norm_2(p_dot)**3)

        return f_t 


    def get_static_full_robot_arm_model(self): 

        return self._static_model_full_robot
    
    def get_robot_arm_params(self): 

        return self._robot_arm_params_obj

    def get_static_robot_arm_model_with_boundaries_model(self): 

        return self._static_model_with_boundaries_

    def get_dynamic_robot_arm_model(self): 

        return self._dynamic_model

    def get_static_robot_arm_model(self): 

        return self._static_model

    def get_boundary_length(self):

        return self._boundary_length

    def set_num_integration_stages(self, stages):

        self._integration_stages = stages

    def set_num_integrator_steps(self, steps): 

        self._integration_steps = steps

    def get_num_integration_stages(self):

        return self._integration_stages

    def get_num_integration_steps(self):

        return self._integration_steps

