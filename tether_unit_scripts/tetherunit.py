import sys
sys.path.insert(0, '../../utils')
import os
import shutil

from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from casadi import *

import numpy as np
from scipy.integrate import odeint
from scipy.optimize._lsq.least_squares import least_squares

import time
import rod_parameterbuilder as rpb

class TetherUnit: 

    def __init__(self, _rod_builder, workspace = None): 

        self._params = _rod_builder.params 
        self._dir_name = 'c_generated_code_' + 'tether_unit'
        if workspace is not None: 
            self._workspace = workspace 
        else: 
            self._workspace = "~"
        self._buildTetherModel()

    def _buildTetherModel(self): 

        self._mass_distribution = self._params['mass_distribution']
        self._Kse = self._params['Kse']
        self._Kbt = self._params['Kbt']
        self._tether_length = self._params['tether_length']
        self._Integrator = None
        self._stepIntegrator = None
        self._integrationSteps = self._params['integration_steps']
        self._robot_type = 'tether_units'

        ### Attritubes for export. ###

        self._dir_list = []
        self._integrator_names = []


        try:

            self._removeOldData()

        except:

            pass

        self._initialiseStates()
        self._createIntegrator()
        self._createstepIntegrator()
        # self._exportData()
        # self._replaceData()

    def _removeOldData(self):

        os.chdir(os.path.expanduser(self._workspace + "/atu_acados_python/lib/shared"))

        os.system("rm *.so")

        self._dir_name = 'c_generated_code_' + self._robot_type
        os.chdir(os.path.expanduser(
            self._workspace + "/atu_acados_python/scripts/" + self._dir_name))

        list_dir = [x[0] for x in os.walk(os.getcwd())]
        list_dir.pop(0)

        if os.getcwd() == os.path.expanduser(self._workspace + "/atu_acados_python/scripts/" + self._dir_name):

            for file_name in os.listdir(os.getcwd()):
                # construct full file path
                file = os.getcwd() + file_name
                if os.path.isfile(file):
                    print('Deleting file:', file)
                    os.remove(file)

            for _dir in list_dir:

                try:

                    print("Deleting " + _dir)
                    shutil.rmtree(_dir)

                except:

                    pass

        else:

            raise ValueError("Removing files in the wrong folder!")

        os.chdir("..")

    def _exportData(self):

        os.chdir(os.path.expanduser(self._workspace + "/atu_acados_python/scripts"))
        
        os.chdir(self._dir_list[0])

        for _dir in self._dir_list:

            os.chdir("../..")
            os.chdir(_dir)
            os.system("mv *.so ../../../../lib/shared")

    def _replaceData(self):

        textToReplace = "struct sim_solver_capsule"
        textToReplace2 = "} sim_solver_capsule"

        for i in range(0, len(self._dir_list)):

            replacedText = textToReplace + self._integrator_names[i]
            replacedText2 = textToReplace2 + self._integrator_names[i]

            os.chdir(os.path.expanduser(
                self._workspace + "/atu_acados_python/scripts/" + self._dir_list[i]))
            fullFileName = "acados_sim_solver_" + self._integrator_names[i]

            # Read in the file
            # with open(fullFileName + '.c', 'r') as file :
            #    filedata = file.read()

            # Replace the target string
            # filedata = filedata.replace(textToReplace, replacedText)

            # Write the file out again
            # with open(fullFileName + '.c', 'w') as file:
            #    file.write(filedata)

            # Read in the file
            with open(fullFileName + '.h', 'r') as file:
                filedata = file.read()

            # Replace the target string
            filedata = filedata.replace(textToReplace, replacedText)
            filedata = filedata.replace(textToReplace2, replacedText2)

            # Write the file out again
            with open(fullFileName + '.h', 'w') as file:
                file.write(filedata)

    def _initialiseStates(self):

        # Initialise all ODE states.

        self._p = SX.sym('p', 3)
        self._eta = SX.sym('self._eta', 4) 
        # self._R = SX.sym('R', 9)
        self._n = SX.sym('n', 3)
        self._m = SX.sym('m', 3)
        self._tau = SX.sym('tau', 1)
        self._alpha = SX.sym('alpha', 1)
        self._curvature = SX.sym('u', 1)

        self._p_d = SX.sym('p_dot', 3)
        self._eta_d = SX.sym('eta_dot', 4)
        self._n_d = SX.sym('n_dot', 3)
        self._m_d = SX.sym('m_dot', 3)
        self._tau_d = SX.sym('tau_dot', 1)
        self._alpha_d = SX.sym('alpha_dot', 1)
        self._Kappa_d = SX.sym('Kappa_d_dot', 1)
        self._curvature_d = SX.sym('u_dot', 1)

        # Initialise constants

        self._g = SX([9.81, 0, 0])
        self._f_ext = self._mass_distribution * self._g
        self._Kappa = SX.sym('Kappa', 1)

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

        # Intermediate states

        self._u = inv(self._Kbt)@transpose(reshape(self._R, 3, 3))@self._m
        self._v = SX([0, 0, 1])
        self._k = 0.1


    def _createIntegrator(self):

        model_name = 'tetherunit_integrator'

        c = self._k*(1-transpose(self._eta)@self._eta)

        u = SX.sym('u')

        p_dot = reshape(self._R, 3, 3) @ self._v
        eta_dot = vertcat(
            0.5*(-self._u[0]*self._eta[1] - self._u[1]*self._eta[2] - self._u[2]*self._eta[3]),
            0.5*(self._u[0]*self._eta[0] + self._u[2]*self._eta[2] - self._u[1]*self._eta[3]),
            0.5*(self._u[1]*self._eta[0] - self._u[2]*self._eta[1] + self._u[0]*self._eta[3]),
            0.5*(self._u[2]*self._eta[0] + self._u[1]*self._eta[1] - self._u[0]*self._eta[2])
        ) + c * self._eta 
        n_dot = - (self._f_ext) + u
        m_dot = - cross(p_dot, self._n) 
        tau_dot = -self._Kappa*self._tau*norm_2(self._u)
        # alpha_dot = 1
        kappa_dot = 0
        # u_dot = 0
        u_dot = norm_2(jacobian(self._u, self._m) @ m_dot + jacobian(self._u, self._eta) @ eta_dot)
        # u_dot = norm_2(inv(-self._Kbt)@((skew(self._u)@self._Kbt@self._u) + skew(self._v)@transpose(self._R)@self._n))

        x = vertcat(self._p, self._eta, self._n, self._m, self._curvature)
        xdot = vertcat(p_dot, eta_dot,
                       n_dot, m_dot, u_dot)
        x_dot_impl = vertcat(self._p_d, self._eta_d, self._n_d, self._m_d, self._curvature_d)
        
        # x = vertcat(self._p, self._eta, self._n, self._m)
        # xdot = vertcat(p_dot, eta_dot,
        #                n_dot, m_dot)
        # x_dot_impl = vertcat(self._p_d, self._eta_d, self._n_d, self._m_d)
        

        self.model = AcadosModel()
        self.model.name = model_name
        self.model.x = x 
        self.model.f_expl_expr = xdot 
        self.model.f_impl_expr = xdot - x_dot_impl
        self.model.u = u
        self.model.z = SX([])
        self.model.xdot = x_dot_impl

        sim = AcadosSim()
        sim.model = self.model 

        Sf = self._tether_length

        os.chdir(os.path.expanduser(
            self._workspace + "/atu_acados_python/scripts/"))

        sim.code_export_directory = self._dir_name + '/' + model_name

        # for exporting data to library folder afterwards.
        self._dir_list.append(sim.code_export_directory)
        self._integrator_names.append(model_name)

        sim.solver_options.T = Sf
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = self._integrationSteps
        # sim.solver_options.sens_forw = False

        acados_integrator = AcadosSimSolver(sim)

        self._Integrator = acados_integrator
        self._linearisedEquations = jacobian(xdot, x)
        self.linearisedEquationsFunction = Function('linear_eq', [x], [self._linearisedEquations])

        return acados_integrator

    def _createstepIntegrator(self):

        model_name = 'tetherunit_stepIntegrator'

        c = self._k*(1-transpose(self._eta)@self._eta)

        p_dot = reshape(self._R, 3, 3) @ self._v
        eta_dot = vertcat(
            0.5*(-self._u[0]*self._eta[1] - self._u[1]*self._eta[2] - self._u[2]*self._eta[3]),
            0.5*(self._u[0]*self._eta[0] + self._u[2]*self._eta[2] - self._u[1]*self._eta[3]),
            0.5*(self._u[1]*self._eta[0] - self._u[2]*self._eta[1] + self._u[0]*self._eta[3]),
            0.5*(self._u[2]*self._eta[0] + self._u[1]*self._eta[1] - self._u[0]*self._eta[2])
        ) + c * self._eta 
        n_dot = - (self._f_ext) 
        m_dot = - cross(p_dot, self._n) 
        tau_dot = -self._Kappa*self._tau*norm_2(self._u)
        alpha_dot = 1
        kappa_dot = 0
        # u_dot = 0
        # u_dot = norm_2(jacobian(self._u, self._m) @ m_dot + jacobian(self._u, self._eta) @ eta_dot)
        u_dot = norm_2(inv(-self._Kbt)@((skew(self._u)@self._Kbt@self._u) + skew(self._v)@transpose(self._R)@self._n))

        x = vertcat(self._p, self._eta, self._n, self._m, self._tau, self._alpha, self._Kappa, self._curvature)
        xdot = vertcat(p_dot, eta_dot,
                       n_dot, m_dot, tau_dot, alpha_dot, kappa_dot, u_dot)
        x_dot_impl = vertcat(self._p_d, self._eta_d, self._n_d, self._m_d, self._tau_d, self._alpha_d, self._Kappa_d, self._curvature_d)

        self.modelStep = AcadosModel()
        self.modelStep.name = model_name
        self.modelStep.x = x 
        self.modelStep.f_expl_expr = xdot 
        self.modelStep.f_impl_expr = xdot - x_dot_impl
        self.modelStep.u = SX([])
        self.modelStep.z = SX([])
        self.modelStep.xdot = x_dot_impl

        sim = AcadosSim()
        sim.model = self.modelStep 

        Sf = self._tether_length/self._integrationSteps

        os.chdir(os.path.expanduser(
            self._workspace + "/atu_acados_python/scripts/"))

        sim.code_export_directory = self._dir_name + '/' + model_name

        # for exporting data to library folder afterwards.
        self._dir_list.append(sim.code_export_directory)
        self._integrator_names.append(model_name)

        sim.solver_options.T = Sf
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 1
        acados_integrator = AcadosSimSolver(sim)

        self._stepIntegrator = acados_integrator

        return acados_integrator