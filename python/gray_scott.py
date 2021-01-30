# File       : gray_scott.py
# Created    : Sat Jan 30 2021 05:12:47 PM (+0100)
# Description: Gray-Scott reaction-diffusion
# Copyright 2021 ETH Zurich. All Rights Reserved.
import numpy as np
import matplotlib.pyplot as plt


class gray_scott:
    """
    Gray-Scott reaction-diffusion system.
    http://mrob.com/pub/comp/xmorphia/index.html

    Reactions:
        A + 2S → 3S
        S → P (P is an inert product)
    """

    def __init__(self,
                 *,
                 F=0.04,
                 kappa=0.06,
                 Da=1.0e-2,
                 Ds=5.0e-3,
                 L=5.5,
                 N=256,
                 Fo=0.9,
                 initial_condition='random'):
        """
        Constructor

        Arguments
            F: parameter F in governing equations
            kappa: parameter kappa (k) in governing equations
            Da: diffusivity of species A
            Ds: diffusivity of species S
            L: domain extent in x and y (square)
            N: number of cells in x and y
            Fo: Fourier number (<= 1)

            initial_condition: type of initial condition to be used
        """
        # parameter
        self.F = F
        self.kappa = kappa
        self.Da = Da
        self.Ds = Ds
        self.L = L
        self.N = N  # cells
        self.V = self.N + 1  # nodes

        # grid spacing
        dx = self.L / self.N

        # intermediates
        self.fa = self.Da / dx**2
        self.fs = self.Ds / dx**2
        self.dt = Fo * dx**2 / (4*max(self.Da, self.Ds))

        # nodal grid
        x = np.linspace(0, self.L, self.V)
        y = np.linspace(0, self.L, self.V)
        self.x, self.y = np.meshgrid(x, y)

        # initial condition
        self.a = np.zeros((self.V, self.V))
        self.s = np.zeros((self.V, self.V))

        if initial_condition == 'random':
            self._random_IC()
        else:
            raise RuntimeError(
                f"Unknown initial condition type: `{initial_condition}`")

        # populate ghost cells
        self.update_ghosts(self.a)
        self.update_ghosts(self.s)

    def update(self):
        """
        Perform Euler integration step
        """
        # internal domain
        a_view = self.a[1:-1, 1:-1]
        s_view = self.s[1:-1, 1:-1]

        # advance state (Euler step)
        as2 = a_view * np.power(s_view, 2)
        a_view += self.dt * (self.fa * self.laplacian(self.a) - as2 + self.F * (1 - a_view))
        s_view += self.dt * (self.fs * self.laplacian(self.s) + as2 - (self.F + self.kappa) * s_view)

        # update ghost cells
        self.update_ghosts(self.a)
        self.update_ghosts(self.s)

    def _random_IC(self):
        """
        Random initial condition
        """
        dim = self.a.shape
        assert dim == self.s.shape
        self.a = np.ones(dim) / 2 + 0.5 * np.random.uniform(0, 1, dim)
        self.s = np.ones(dim) / 4 + 0.5 * np.random.uniform(0, 1, dim)

    @staticmethod
    def laplacian(a):
        """
        Discretization of Laplacian operator
        """
        return a[2:, 1:-1] + a[1:-1, 2:] + a[0:-2, 1:-1] + a[1:-1, 0:-2] - 4 * a[1:-1, 1:-1]

    @staticmethod
    def update_ghosts(v):
        """
        """
        v[0, :] = v[-2, :]
        v[:, 0] = v[:, -2]
        v[-1, :] = v[1, :]
        v[:, -1] = v[:, 1]
