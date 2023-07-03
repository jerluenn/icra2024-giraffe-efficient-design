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

class Rod_Parameter_Builder:

    def __init__(self): 

        self._valid_build = False

    def createCustomParameters(self, params):

        if not self._valid_build: 

            print("Building Continuum Robot using Custom Parameters...")

            self.params = params
            self.physical_type = 'Custom'
            self.params['physical_type'] = self.physical_type

            self._checkDict(self.params)

        else: 

            raise ValueError("Parameters have been built before.")

        return self.params

    def createHollowRod(self, params):


        if not self._valid_build: 

            print("Building Continuum Robot with Hollow Rod...")

            self.params = params
            self._ir = params['inner_radius']
            self._or = params['outer_radius']
            self._E = params['elastic_modulus']
            self._shear_mod = params['shear_modulus']
            try:
                self._rho = params['rho']
                self.mass_distribution = self._rho * self._area
            except:
                self.mass_distribution = params['mass_distribution']
            self._area = np.pi * self._or**2 - np.pi * self._ir**2
            self._I = ((np.pi * self._or**4) / 4) - \
                ((np.pi * self._ir**4) / 4)
            self._J = 2 * self._I
            self.Kse = diag([self._shear_mod * self._area,
                            self._shear_mod * self._area, self._E * self._area])
            self.Kbt = diag([self._E * self._I, self._E *
                            self._I, self._shear_mod * self._J])
            self.physical_type = 'hollow_rod'
            self.params['physical_type'] = self.physical_type
            self.params['Kse'] = self.Kse
            self.params['Kbt'] = self.Kbt
            self.params['mass_distribution'] = self.mass_distribution

            self._checkDict(self.params)

        else: 

            raise ValueError("Parameters have been built before.")


        return self.params

    def createRod(self, params):

        if not self._valid_build: 

            print("Building Continuum Robot with Solid Rod...")

            self.params = params
            self._r = params['radius']
            self._E = params['elastic_modulus']
            self._shear_mod = params['shear_modulus']
            try:
                self._rho = params['rho']
                self.mass_distribution = self._rho * self._area
            except:
                self.mass_distribution = params['mass_distribution']
            self._area = np.pi * self._r**2
            self._I = (np.pi * self._r**4) / 4
            self._J = 2 * self._I
            self.Kse = diag([self._shear_mod * self._area,
                            self._shear_mod * self._area, self._E * self._area])
            self.Kbt = diag([self._E * self._I, self._E *
                            self._I, self._shear_mod * self._J])
            self.mass_distribution = self._rho * self._area
            self.physical_type = 'Solid Rod'
            self.params['physical_type'] = self.physical_type
            self.params['Kse'] = self.Kse
            self.params['Kbt'] = self.Kbt
            self.params['mass_distribution'] = self.mass_distribution

            self._checkDict(self.params)

        else: 

            raise ValueError("Parameters have been built before.")



        return self.params

    def _checkDict(self, params):

        try:

            assert type(params['mass_distribution']) == float, "mass_distribution must be a float."
            assert type(params['Kse']) == DM, "Kse must be of DM type."
            assert type(params['Kbt']) == DM, "Kbt must be of DM type."
            assert type(params['tether_length']) == float, "tether_length must be a float."
            self._valid_build = True
            print("Valid build.")

        except:

            raise ValueError(
                "Building failed! Params must include mass distribution, tip weight, Kse, Kbt, tether length.")