from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ylim = 5
alpha = 1
X = np.linspace(-50, 50, 200)

def f(x, t):
    return 0.5*(erf((ylim-x)/(np.sqrt(alpha*t)))-erf((-ylim-x)/(np.sqrt(alpha*t))))

fig = plt.figure()

def animate(t):
    print(f"Frame: {t+1}")
    plt.clf()
    Y = f(X, t+1e-10)
    plt.plot(X, Y, color="gray")
    plt.ylim(0, 1)
    plt.title(f"Unit time: {t}")
    plt.xlabel("x")
    plt.ylabel("u(x)")

anim = FuncAnimation(fig, animate, frames = 500, interval = 1)
anim.save("Wave1D.gif", fps = 30, dpi=200)