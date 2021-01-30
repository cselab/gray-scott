#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from gray_scott import gray_scott

gs = gray_scott()

T_frames = 100
T_between_frames = 10
A = np.zeros((T_frames+1,gs.Nx,gs.Ny))
A[0,:,:] = gs.a

print('Solve the Gray-Scott model...\n')

for k in range(T_frames):
    for l in range(T_between_frames):
        gs.update()
    A[k+1,:,:] = gs.a

print('Finished solving the Gray-Scott model...\n')



class video_plot(object):
    def __init__(self, ax, skip_frames=10):
        self.ax = ax
        self.cntr = []

    def __call__(self, i):
        try:
            for c in self.cntr.collections:
                c.remove()
        except:
            pass

        self.cntr = self.ax.contourf( gs.x, gs.y, A[i,:,:], levels=50, cmap=plt.cm.gray )
        self.ax.set_title( str(i*T_between_frames) )
        return self.cntr


fig, ax = plt.subplots(1, 1, figsize=(10,10))

up = video_plot( ax , skip_frames=0)

anim = FuncAnimation( fig, up, frames=range(T_frames+1), interval=1, blit=False, repeat=False )

plt.show()

writer = PillowWriter(fps=25)
anim.save("gs.gif", writer=writer)
