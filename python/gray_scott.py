# File       : gray_scott.py
# Created    : Sat Jan 30 2021 05:12:47 PM (+0100)
# Description: Gray-Scott reaction-diffusion
# Copyright 2021 ETH Zurich. All Rights Reserved.
import os
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt

class GrayScott:
    """
    Gray-Scott reaction-diffusion system.
    http://mrob.com/pub/comp/xmorphia/index.html
    https://www.lanevol.org/resources/gray-scott

    Reactions:
        A + 2S → 3S
        S → P (P is an inert product)
    """

    def __init__(self,
                 *,
                 F=0.04,
                 kappa=0.06,
                 Du=2.0e-5,
                 Dv=1.0e-5,
                 x0=-1,
                 x1=1,
                 N=256,
                 Fo=0.8,
                 initial_condition='trefethen',
                 second_order=False,
                 movie=False,
                 outdir='.'):
        """
        Constructor.
        The domain is a square.

        Arguments
            F: parameter F in governing equations
            kappa: parameter kappa (k) in governing equations
            Du: diffusivity of species U
            Dv: diffusivity of species V
            x0: left domain coordinate
            x1: right domain coordinate
            N: number of nodes in x and y
            Fo: Fourier number (<= 1)

            initial_condition: type of initial condition to be used
            second_order: use Heun's method (2nd-order Runge-Kutta)
            movie: create a movie
            outdir: output directory
        """
        # parameter
        self.F = F
        self.kappa = kappa

        Nnodes = N + 1
        self.Fo = Fo
        self.x0 = x0
        self.x1 = x1

        # options
        self.movie = movie
        self.outdir = outdir
        self.dump_count = 0
        self.second_order = second_order

        # grid spacing
        L = x1 - x0
        dx = L / N

        # intermediates
        self.fa = Du / dx**2
        self.fs = Dv / dx**2
        self.dt = Fo * dx**2 / (4*max(Du, Dv))

        # nodal grid (+ghosts)
        x = np.linspace(x0-dx, x1+dx, Nnodes+2)
        y = np.linspace(x0-dx, x1+dx, Nnodes+2)
        self.x, self.y = np.meshgrid(x, y)

        # initial condition
        self.u = np.zeros((len(x), len(y)))
        self.v = np.zeros((len(x), len(y)))
        if self.second_order:
            self.u_tmp = np.zeros((len(x), len(y)))
            self.v_tmp = np.zeros((len(x), len(y)))

        if initial_condition == 'trefethen':
            self._trefethen_IC()
        elif initial_condition == 'random':
            self._random_IC()
        else:
            raise RuntimeError(
                f"Unknown initial condition type: `{initial_condition}`")

        # populate ghost cells
        self.update_ghosts(self.u)
        self.update_ghosts(self.v)

    def integrate(self, t0, t1, *, dump_freq=100, report=50):
        """
        Integrate system.

        Arguments:
            t0: start time
            r1: end time
            dump_freq: dump frequency in steps
            report: stdout report frequency
        """
        t = t0
        s = 0
        while t < t1:
            if s % dump_freq == 0:
                self._dump(s, t)
            if s % report == 0:
                print(f"step={s}; time={t:e}")
            t = self.update(time=t)
            if (t1 - t) < self.dt:
                self.dt = t1 - t
            s += 1

        if self.movie:
            self._render_frames()


    def update(self, *, time=0):
        """
        Perform time integration step

        Arguments:
            time: current time

        Returns:
            Time after integration
        """
        if self.second_order:
            self._heun()
        else:
            self._euler()

        # update ghost cells
        self.update_ghosts(self.u)
        self.update_ghosts(self.v)

        return time + self.dt


    def _euler(self):
        """
        1st order Euler step
        """
        # internal domain
        u_view = self.u[1:-1, 1:-1]
        v_view = self.v[1:-1, 1:-1]

        # advance state (Euler step)
        uv2 = u_view * np.power(v_view, 2)
        u_view += self.dt * (self.fa * self.laplacian(self.u) - uv2 + self.F * (1 - u_view))
        v_view += self.dt * (self.fs * self.laplacian(self.v) + uv2 - (self.F + self.kappa) * v_view)


    def _heun(self):
        """
        2nd order Heun's method
        """
        # internal domain
        u_view = self.u[1:-1, 1:-1]
        v_view = self.v[1:-1, 1:-1]
        u_vtmp = self.u_tmp[1:-1, 1:-1]
        v_vtmp = self.v_tmp[1:-1, 1:-1]

        # 1st stage
        uv2 = u_view * v_view**2
        u_rhs1 = self.fa * self.laplacian(self.u) - uv2 + self.F * (1 - u_view)
        v_rhs1 = self.fs * self.laplacian(self.v) + uv2 - (self.F + self.kappa) * v_view
        u_vtmp = u_view + self.dt * u_rhs1
        v_vtmp = v_view + self.dt * v_rhs1
        self.update_ghosts(self.u_tmp)
        self.update_ghosts(self.v_tmp)

        # 2nd stage
        uv2 = u_vtmp * v_vtmp**2
        u_rhs2 = self.fa * self.laplacian(self.u_tmp) - uv2 + self.F * (1 - u_vtmp)
        v_rhs2 = self.fs * self.laplacian(self.v_tmp) + uv2 - (self.F + self.kappa) * v_vtmp
        u_view += 0.5 * self.dt * (u_rhs1 + u_rhs2)
        v_view += 0.5 * self.dt * (v_rhs1 + v_rhs2)


    def _dump(self, step, time, *, both=False):
        """
        Dump snapshot

        Arguments:
            step: step ID
            time: current time
        """
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        x = self.x[1:-1, 1:-1]
        y = self.y[1:-1, 1:-1]
        U = self.u[1:-1, 1:-1]
        V = self.v[1:-1, 1:-1]
        if both:
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            cs0 = ax[0].contourf(x, y, U, levels=50, cmap='jet')
            cs1 = ax[1].contourf(x, y, V, levels=50, cmap='jet')
            fig.suptitle(f"time = {time:e}")
            lim = (self.x0, self.x1)
            species = ("U", "V")
            for a, l in zip(ax, species):
                a.set_title(f"Species {l}")
                a.set_xlabel("x")
                a.set_ylabel("y")
                a.set_xlim(lim)
                a.set_ylim(lim)
                a.set_aspect('equal')
        else: # only species V
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.contourf(x, y, V, levels=50, cmap='jet')
            fig.suptitle(f"time = {time:e}")
            lim = (self.x0, self.x1)
            ax.set_title(f"Species V")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
        fig.savefig(os.path.join(self.outdir, f"frame_{self.dump_count:06d}.png"), dpi=400)
        plt.close(fig)
        self.dump_count += 1


    def _render_frames(self):
        cmd = ['ffmpeg', '-framerate', '24', '-i', 
                os.path.join(self.outdir, 'frame_%06d.png'), '-b:v', '90M', 
                '-vcodec', 'mpeg4', os.path.join(self.outdir, 'movie.mp4')]
        sp.run(cmd)


    def _random_IC(self):
        """
        Random initial condition
        """
        dim = self.u.shape
        self.u = np.ones(dim) / 2 + 0.5 * np.random.uniform(0, 1, dim)
        self.v = np.ones(dim) / 4 + 0.5 * np.random.uniform(0, 1, dim)

    def _trefethen_IC(self):
        """
        Initial condition for the example used by N. Trefethen at
        https://www.chebfun.org/examples/pde/GrayScott.html
        """
        x = self.x[1:-1, 1:-1]
        y = self.y[1:-1, 1:-1]
        self.u[1:-1, 1:-1] = 1 - np.exp(-80*((x+0.05)**2 + (y+0.05)**2))
        self.v[1:-1, 1:-1] = np.exp(-80*((x-0.05)**2 + (y-0.05)**2))


    @staticmethod
    def laplacian(a):
        """
        Discretization of Laplacian operator

        Arguments:
            a: 2D array
        """
        return a[2:, 1:-1] + a[1:-1, 2:] + a[0:-2, 1:-1] + a[1:-1, 0:-2] - 4 * a[1:-1, 1:-1]

    @staticmethod
    def update_ghosts(v):
        """
        Apply periodic boundary conditions

        Arguments:
            v: 2D array
        """
        v[0, :] = v[-2, :]
        v[:, 0] = v[:, -2]
        v[-1, :] = v[1, :]
        v[:, -1] = v[:, 1]
