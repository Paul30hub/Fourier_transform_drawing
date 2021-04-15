#%%
from PIL import Image
from pylab import *

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
from numpy import interp

from math import tau
from scipy.integrate import quad

#%%
def create_close_loop(image_name, level=[200]):
    # Prepare Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_aspect('equal', 'datalim')
    ax[1].set_aspect('equal', 'datalim')
    ax[0].set_title('Before Centered')
    ax[1].set_title('After Centered')

    # read image to array, then get image border with contour
    im = array(Image.open(image_name).convert('L'))
    contour_plot = ax[0].contour(im, levels=level, colors='black', origin='image')

    # Get Contour Path and create lookup-table
    contour_path = contour_plot.collections[0].get_paths()[0]
    x_table, y_table = contour_path.vertices[:, 0], contour_path.vertices[:, 1]
    time_table = np.linspace(0, tau, len(x_table))

    # Simple method to center the image
    x_table = x_table - min(x_table)
    y_table = y_table - min(y_table)
    x_table = x_table - max(x_table) / 2
    y_table = y_table - max(y_table) / 2

    # Visualization
    ax[1].plot(x_table, y_table, 'k-')

    return time_table, x_table, y_table

def f(t, time_table, x_table, y_table):
    return interp(t, time_table, x_table) + 1j*interp(t, time_table, y_table)