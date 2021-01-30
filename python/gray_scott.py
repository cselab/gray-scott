import numpy as np
import matplotlib.pyplot as plt


def initial_condition(Nx, Ny):
  a = np.ones((Nx,Ny))/2 + 0.5*np.random.uniform(0,1,(Nx,Ny));
  s = np.ones((Nx,Ny))/4 + 0.5*np.random.uniform(0,1,(Nx,Ny));
  return a, s

def update_ghosts(v):
    v[0,:] = v[-2,:];
    v[:,0] = v[:,-2];
    v[-1,:] = v[1,:];
    v[:,-1] = v[:,1];

def laplacian(a):
  return a[2:,1:-1] + a[1:-1,2:] + a[0:-2,1:-1]  + a[1:-1,0:-2] - 4*a[1:-1,1:-1]


class grey_scott:

  def __init__(self):
    self.F= 0.04;
    self.kappa = 0.06;
    self.D_a = 1e-2;
    self.D_s = 5e-3;

    self.fq = 0.90;

    self.Lx = 5.50;
    self.Ly = 5.50;
    self.Nx = 257;
    self.Ny = 257;
    dx = self.Lx/(self.Nx-1);
    dy = self.Ly/(self.Ny-1);

    self.fa = self.D_a/dx**2;
    self.fs = self.D_s/dx**2;
    self.dt = self.fq/(4.0*self.fa);

    x = np.linspace(0, self.Lx, self.Nx)
    y = np.linspace(0, self.Ly, self.Ny)
    self.x, self.y = np.meshgrid(x, y)


    [self.a,self.s] = initial_condition(self.Nx,self.Ny);

    update_ghosts(self.a);
    update_ghosts(self.s);

  def update(self):
    as2 = self.a[1:-1,1:-1]*np.power(self.s[1:-1,1:-1],2)

    self.a[1:-1,1:-1] += self.dt*( self.fa*laplacian(self.a) - as2 + self.F*(1-self.a[1:-1,1:-1]) )
    self.s[1:-1,1:-1] += self.dt*( self.fs*laplacian(self.s) + as2 - (self.F+self.kappa)*self.s[1:-1,1:-1] )

    update_ghosts(self.a);
    update_ghosts(self.s);


