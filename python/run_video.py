#! /usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import sys

from grey_scott import grey_scott


class video_plot(object):
  def __init__(self, ax, skip_frames=10):
    self.ax = ax
    self.gs = grey_scott()
    self.skip_frames = skip_frames
    if(skip_frames<0):
      sys.exit("skip_frames must be >= 0")

  def __call__(self, i):
    if i > 0:
      for k in range(self.skip_frames+1):
        self.gs.update()

    cntr = self.ax.contourf( self.gs.x, self.gs.y, self.gs.a )
    self.ax.set_title( str(i*(self.skip_frames+1)) )
    return cntr


Tmax = 10

fig, ax = plt.subplots(1, 1, figsize=(10,10))

up = video_plot( ax , skip_frames=0)

anim = FuncAnimation( fig, up, frames=Tmax, interval=1, blit=False, repeat=False )

plt.show()

writer = PillowWriter(fps=25)
anim.save("gs.gif", writer=writer)