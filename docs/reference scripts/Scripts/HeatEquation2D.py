from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

u0 = 100
rmax = 5
alpha = 1
rx0 = 0
ry0 = 0

def f(X, Y, t):
    rr = np.sqrt((X-rx0)**2 + (Y-ry0)**2)
    return 0.5*u0*(erf((rmax-rr)/(np.sqrt(alpha*t)))-erf((-rmax-rr)/(np.sqrt(alpha*t))))

P = 200
Xlist = np.linspace(-50, 50, P)
Ylist = np.linspace(-50, 50, P)
X, Y = np.meshgrid(Xlist, Ylist)

fig = plt.figure()

def animate(t):
    print(f"Frame: {t+1}")
    plt.clf()
    R = f(X, Y, t+1e-10)
    plt.imshow(R, cmap="jet", interpolation="bilinear", extent=(-50, 50, -50, 50), vmin=0, vmax=100)
    plt.colorbar(label="Temperature [K]")
    plt.title(f"Unit time: {t}")
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    
anim = FuncAnimation(fig, animate, frames = 500, interval = 1, repeat=False)
anim.save("Wave2D.gif", fps = 30, dpi=200)