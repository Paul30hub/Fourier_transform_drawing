import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from matplotlib.patches import ConnectionPatch
from math import tau
import cmath

class DrawAnimation:
    """Creates an animation that draws a curve representing an image using Fourier coefficients.
    
    :param x_DFT: Real part of the Fourier transform.
    :type x_DFT: list.
    :param y_DFT: Imaginary part of the Fourier transform.
    :type y_DFT: list.
    :param coef: Fourier coefficients.
    :type coef: ndarray.
    :param order: Variable that we use to determine the number of Fourier coefficients that we will generate.
    :type order: int.
    :param space: Variable used to give the number of images for the animation
    :type space: tuple.
    :param fig_lim: Define the limit of the x and y axes of our plot.
    :type fig_lim: list.
    """
    def __init__(self, x_DFT, y_DFT, coef, order, space, fig_lim):
        """
            Construction method
        """
        self.x_DFT = x_DFT
        self.y_DFT = y_DFT
        self.coef = coef 
        self.order = order
        self.space = space
        self.fig_lim = fig_lim
        
    def visualize(self, x_DFT, y_DFT, coef, order, space, fig_lim):
        """
            Creates the drawing of the image and spinning circles animation.
        """
        fig, ax = plt.subplots()
        lim = max(fig_lim)
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_aspect('equal')

        # Initialize
        line = plt.plot([], [], 'k-', linewidth = 2)[0]
        radius = [plt.plot([], [], 'r-', linewidth = 0.5, marker = 'o', markersize = 1)[0] for _ in range(2 * order + 1)]
        circles = [plt.plot([], [], 'r-', linewidth = 0.5)[0] for _ in range(2 * order + 1)]

        def update_c(c, t):
            """
                Creates a new ndarray new_c for plot radius in each circle and plot each circle in the 
                function animate.
            """
            new_c = []
            for i, j in enumerate(range(-order, order + 1)):
                dtheta = -j * t
                ct, st = np.cos(dtheta), np.sin(dtheta)
                v = [ct * c[i][0] - st * c[i][1], st * c[i][0] + ct * c[i][1]]
                new_c.append(v)
            return np.array(new_c)

        def sort_velocity(order):
            """
                Creates a variable that iterates through the numbers between order+i and order-i.
            """
            idx = []
            for i in range(1, order+1):
                idx.extend([order+i, order-i]) 
            return idx    

        def animate(i):
            # animate lines
            """ 
                Displays the radius of each circle as well as the circle. This function also plots the approximation of the image.
            """
            line.set_data(x_DFT[:i], y_DFT[:i])
            # array of radius of each circles
            r = [np.linalg.norm(coef[j]) for j in range(len(coef))]
            # position is on the last circle
            pos = coef[order]
            c = update_c(coef, i / len(space) * tau)
            idx = sort_velocity(order)
            for j, rad, circle in zip(idx, radius, circles):
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
        ani = animation.FuncAnimation(fig, animate, frames = len(space), interval=5)
        return ani
