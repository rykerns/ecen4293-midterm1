from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ts = 2e15

R1 = 6400e3
R2 = 6371e3
R3 = 3486e3
R4 = 1216e3

def f(X, Y, A, t, rx0, ry0, u0, rmax):
    rr = np.sqrt((X-rx0)**2 + (Y-ry0)**2)
    return 0.5*u0*(erf((rmax-rr)/(np.sqrt(A*t)))-erf((-rmax-rr)/(np.sqrt(A*t))))

P = 200
Xlist = np.linspace(0, 6400e3, P)
Ylist = np.linspace(0, 6400e3, P)
X, Y = np.meshgrid(Xlist, Ylist)

A = np.ones((P,P))
A[True] = 1e-50
A[np.sqrt(X**2 + (Y-6400e3)**2) <= R1] = 1e-5
A[np.sqrt(X**2 + (Y-6400e3)**2) <= R2] = 1e-6
A[np.sqrt(X**2 + (Y-6400e3)**2) <= R3] = 1e-7
A[np.sqrt(X**2 + (Y-6400e3)**2) <= R4] = 1e-8

#Layered Earth Approach:
#E1 = np.zeros((P,P))
#E1[np.sqrt(X**2 + (Y-6400e3)**2) <= R1] = 300
#E1[np.sqrt(X**2 + (Y-6400e3)**2) <= R2] = 2300
#E1[np.sqrt(X**2 + (Y-6400e3)**2) <= R3] = 4100
#E1[np.sqrt(X**2 + (Y-6400e3)**2) <= R4] = 6300

fig = plt.figure()

def animate(t):
    print(f"Frame: {t+1}")
    plt.clf()
    E0 = f(X, Y, A, ts*t+1e-10, 4500e3, 1900e3, 1e8, 1e2)
    E1 = f(X, Y, A, ts*t+2e20, 0, 6400e3, 8000, R4)
    plt.imshow(E0+E1, cmap="jet", interpolation="bilinear", vmin=0, vmax=1e4, extent=(0, 6400e3, 0, 6400e3))
    plt.colorbar(label="Temperature [K]")
    plt.title(f"Elapsed time [s]: {t*2}"+"$\cdot10^{15}$")
    plt.xlabel("x axis [m]")
    plt.ylabel("y axis [m]")
    
anim = FuncAnimation(fig, animate, frames = 100, interval = 1, repeat=False)
anim.save("WaveComet.gif", fps = 30, dpi=200)