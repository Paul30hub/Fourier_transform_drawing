import matplotlib.pyplot as plt
import numpy as np
from math import tau
from scipy.integrate import quad
from numpy import interp

def f(t, time_table, x_table, y_table):
            """
            Convert the X and Y coords to complex number over time
            """
            X = np.interp(t, time_table, x_table) 
            Y = 1j * np.interp(t, time_table, y_table)
            return X + Y

class Fourier :
    """
        Convert coordinates found by ImageReader class to complex numbers with real and imaginary part.
    
    :param time_table: Time list from 0 to 2PI.
    :type time_table: tuple
    :param x_table: Coordinate of the X axis.
    :type x_table: float
    :param y_table: Coordinate on the Y axis
    :type y_table: float
    :param order: Variable that we use to determine the number of Fourier coefficients that we will generate.
    :type order: int
    """

    def __init__(self, time_table, x_table, y_table, order) :
        self.order = order
        self.time_table = time_table
        self.x_table = x_table
        self.y_table = y_table

    def coef_list(self,time_table, x_table, y_table, order) :
        coef_list = []
            
        for n in range(-order, order+1):
        #integrate across f .
            real_coef = quad(lambda t: np.real(f(t, time_table, x_table, y_table) * np.exp(-n * 1j * t)), 0, tau, limit=100, full_output=1)[0] / tau
            imag_coef = quad(lambda t: np.imag(f(t, time_table, x_table, y_table) * np.exp(-n * 1j * t)), 0, tau, limit=100, full_output=1)[0] / tau
            coef_list.append([real_coef, imag_coef])

        return np.array(coef_list)


    def DFT(self, t, coef_list, order):
    #Compute the discrete fourier series with a given order
        kernel = np.array([np.exp(-n * 1j * t) for n in range(-order, order+1)])
        series = np.sum( (coef_list[:,0] + 1j * coef_list[:,1]) * kernel[:])

        return np.real(series), np.imag(series)
