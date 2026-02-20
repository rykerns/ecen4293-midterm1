from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

P = 2000
NrFr = 1000
CoNr = 25
delay = 20
crater = 100
inter="bilinear"
dpi=200
ts = 2e16

R1 = 6400e3
R2 = 6371e3
R3 = 3486e3
R4 = 1216e3

Lpx = []
Lpy = []
Ltemp = []
Lt = []
Ltk = []

def newComet():
    temp = random.randint(8e4, 5e5)
    angle = random.randint(0, 62831853071)/1e10
    px = (R1-1e5)*np.sin(angle) + 0.5*k
    py = (R1-1e5)*np.cos(angle) + 0.5*k

    print("angle: ", angle)
    print("px: ", px)
    print("py: ", py)
    print()
    return px, py, temp

def f(X, Y, A, t, rx0, ry0, u0, rmax, init):
    if t>=init:
        t=t-init+1e-10
        rr = np.sqrt((X-rx0)**2 + (Y-ry0)**2)
        return 0.5*u0*(erf((rmax-rr)/(np.sqrt(A*t)))-erf((-rmax-rr)/(np.sqrt(A*t))))
    else: return 0

def impact(t):
    TL = []
    for i in range(len(Lt)):
        TL.append(f(X, Y, A, ts*t+1e-10, Lpx[i], Lpy[i], Ltemp[i]*crater, 1e2, Lt[i]*ts))
    return sum(TL)

k = 1.5e7
Xlist = np.linspace(0, k, P)
Ylist = np.linspace(0, k, P)
X, Y = np.meshgrid(Xlist, Ylist)

V = np.zeros((P,P))
Zs = np.zeros((P,P))

A = np.ones((P,P))
A[True] = 1e-100
A[np.sqrt((X-0.5*k)**2 + (Y-0.5*k)**2) <= R1] = 1e-5
A[np.sqrt((X-0.5*k)**2 + (Y-0.5*k)**2) <= R2] = 1e-6
A[np.sqrt((X-0.5*k)**2 + (Y-0.5*k)**2) <= R3] = 1e-7
A[np.sqrt((X-0.5*k)**2 + (Y-0.5*k)**2) <= R4] = 1e-8

fig = plt.figure()

def animate(t):
    print(f"Frame: {t}")
    global tk
    if t==0:
        tk=-1
        
    if (t%delay)==0 and len(Lt)<CoNr and t>0:
        global px, py, temp
        px, py, temp = newComet()
        
        Lpx.append(px)
        Lpy.append(py)
        Ltemp.append(temp)
        Ltk.append(tk)
        Lt.append(t)
    
    tk +=1
    plt.clf()
    E0 = impact(t)
    E1 = f(X, Y, A, ts*t+1e-10, 0.5*k, 0.5*k, 10000, R4, -3e20)
    plt.imshow(E0+E1, cmap="jet", interpolation=inter, vmin=0, vmax=1e4, extent=(-0.5*k, 0.5*k, -0.5*k, 0.5*k), origin="lower")
    plt.colorbar(label="Temperature [K]")
    plt.title(f"Elapsed time [s]: {t*2}"+"$\cdot10^{16}$")
    plt.xlabel("x axis [m]")
    plt.ylabel("y axis [m]")

anim = FuncAnimation(fig, animate, frames = NrFr, interval = 1, repeat=False)
anim.save(f"Frames {NrFr}, P {P}, CoNr {CoNr}.gif", fps = 30, dpi=dpi)
