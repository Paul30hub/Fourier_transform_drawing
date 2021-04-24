import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from matplotlib.patches import ConnectionPatch
from math import tau
import cmath

class DrawAnimation:
    def __init__(self, x_DFT, y_DFT, coef, order, space, fig_lim):
        self.x_DFT = x_DFT
        self.y_DFT = y_DFT
        self.coef = coef 
        self.order = order
        self.space = space
        self.fig_lim = fig_lim
        
    def visualize(self, x_DFT, y_DFT, coef, order, space, fig_lim):
        fig, ax = plt.subplots()
        lim = max(fig_lim)
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_aspect('equal')

        # Initialize
        line = plt.plot([], [], 'k-', linewidth=2)[0]
        radius = [plt.plot([], [], 'r-', linewidth=0.5, marker='o', markersize=1)[0] for _ in range(2 * order + 1)]
        circles = [plt.plot([], [], 'r-', linewidth=0.5)[0] for _ in range(2 * order + 1)]

        def update_c(c, t):
            new_c = []
            for i, j in enumerate(range(-order, order + 1)):
                dtheta = -j * t
                ct, st = np.cos(dtheta), np.sin(dtheta)
                v = [ct * c[i][0] - st * c[i][1], st * c[i][0] + ct * c[i][1]]
                new_c.append(v)
            return np.array(new_c)

        def sort_velocity(order):
            idx = []
            for i in range(1,order+1):
                idx.extend([order+i, order-i]) 
            return idx    

        def animate(i):
            # animate lines
            """ 
                Create the drawing of the image and spinning circles sanimation
            """
            line.set_data(x_DFT[:i], y_DFT[:i])
            # array of radius of each circles
            r = [np.linalg.norm(coef[j]) for j in range(len(coef))]
            # position is on the last circle
            pos = coef[order]
            c = update_c(coef, i / len(space) * tau)
            idx = sort_velocity(order)
            for j, rad, circle in zip(idx,radius,circles):
                new_pos = pos + c[j]
                # plot radius in each circles
                rad.set_data([pos[0], new_pos[0]], [pos[1], new_pos[1]])
                theta = np.linspace(0, tau, 50)
                # plot each circles
                x, y = r[j] * np.cos(theta) + pos[0], r[j] * np.sin(theta) + pos[1]
                circle.set_data(x, y)
                # increase pos for plot from the last circle displayed
                pos = new_pos

        # Animation
        ani = animation.FuncAnimation(fig, animate, frames=len(space), interval=5)
        return ani
