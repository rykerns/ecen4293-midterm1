from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

P = 200
NrFr = 300
CoNr = 100
delay = 100
crater = 20
inter="none"
dpi=100
ts = 2e16
corrconst = 0.5

R1 = 6400e3
R2 = 6371e3
R3 = 3486e3
R4 = 1216e3

Lpx = []
Lpy = []
Lsx = []
Lsy = []
Lsize = []
Lt = []
Ltk = []

def newComet():
    size = random.randint(8e4, 5e5)
    speed = 5e5
    p0 = random.randint(0, 1)
    if p0==0:
        px=random.randint(500e3, 10100e3)
        py=-100e3
        sx=(px-500e3)/(9400e3)
        sy=1
    else:
        px=10100e3
        py=random.randint(0, 9400e3)
        sx=1
        sy=1-(py/(9400e3))
    print("p0: ", p0)
    print("px: ", px)
    print("py: ", py)
    print("sx: ", sx)
    print("sy: ", sy)
    print()
    return px, py, sx, sy, speed, size

def f(X, Y, A, t, rx0, ry0, u0, rmax, init):
    if t>=init:
        t=t-init+1e-10
        rr = np.sqrt((X-rx0)**2 + (Y-ry0)**2)
        return 0.5*u0*(erf((rmax-rr)/(np.sqrt(A*t)))-erf((-rmax-rr)/(np.sqrt(A*t))))
    else: return 0

def g(X, Y, t, px, py, sx, sy, speed, size):
    V[True] = 0
    V[np.sqrt((X-(px-speed*t*sx))**2 + (Y-(py+speed*t*sy))**2) < size] = 10000
    return V

def impact(t):
    TL = []
    for i in range(len(Lt)):
        corr = corrconst*Lsize[i]
        TL.append(f(X, Y, A, ts*t+1e-10, Lpx[i]-(speed*Ltk[i]*Lsx[i]+corr*Lsx[i]), Lpy[i]+(speed*Ltk[i]*Lsy[i]+corr*Lsy[i]), Lsize[i]*crater, 1e2, Lt[i]*ts))
    return sum(TL)

k = 1e7
Xlist = np.linspace(0, k, P)
Ylist = np.linspace(0, k, P)
X, Y = np.meshgrid(Xlist, Ylist)

V = np.zeros((P,P))
Zs = np.zeros((P,P))

A = np.ones((P,P))
A[True] = 1e-100
A[np.sqrt(X**2 + (Y-k)**2) <= R1] = 1e-5
A[np.sqrt(X**2 + (Y-k)**2) <= R2] = 1e-6
A[np.sqrt(X**2 + (Y-k)**2) <= R3] = 1e-7
A[np.sqrt(X**2 + (Y-k)**2) <= R4] = 1e-8

fig = plt.figure()

def animate(t):
    print(f"Frame: {t}")

    if (t%delay)==0 and len(Lt)<CoNr:
        global tk
        tk=-1
        global px, py, sx, sy, speed, size
        px, py, sx, sy, speed, size = newComet()
    
    tk +=1
    plt.clf()
    E0 = impact(t)
    E1 = f(X, Y, A, ts*t+1e-10, 0, k, 10000, R4, -3e20)
    G = g(X, Y, tk, px, py, sx, sy, speed, size)
    
    C0 = np.ones((P,P))
    C0[True] = 10000
    C1=((E1+G)-C0)
    C2=(abs(C1)+C1)/2
    C3=sum(sum(C2))
    
    if C3<1000:
        global saveImpact
        saveImpact=False
        plt.imshow(E0+E1+G, cmap="jet", interpolation=inter, vmin=0, vmax=1e4, extent=(0, k, 0, k))
    else:
        if saveImpact==False:
            saveImpact=True
            Lpx.append(px)
            Lpy.append(py)
            Lsx.append(sx)
            Lsy.append(sy)
            Lsize.append(size)
            Ltk.append(tk)
            Lt.append(t)
            print("Impact:", len(Lt))
            print()
        plt.imshow(E0+E1, cmap="jet", interpolation=inter, vmin=0, vmax=1e4, extent=(0, k, 0, k))
        
    plt.colorbar(label="Temperature [K]")
    plt.title(f"Elapsed time [s]: {t*2}"+"$\cdot10^{16}$")
    plt.xlabel("x axis [m]")
    plt.ylabel("y axis [m]")

anim = FuncAnimation(fig, animate, frames = NrFr, interval = 1, repeat=False)
anim.save(f"Frames {NrFr}, P {P}, CoNr {CoNr}.gif", fps = 30, dpi=dpi)
