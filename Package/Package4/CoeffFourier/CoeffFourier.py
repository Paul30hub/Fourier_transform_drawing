import matplotlib.pyplot as plt
import numpy as np
from math import tau
from scipy.integrate import quad
from numpy import interp


class Fourier : 

    def __init__(self, time_table, x_table, y_table, order) :
        self.order = order
        self.time_table = time_table
        self.x_table = x_table
        self.y_table = y_table

    #@staticmethod

    def coef_list(self,time_table, x_table, y_table, order) :

        def func(t, time_table, x_table, y_table):
            return np.interp(t, time_table, x_table + 1j*y_table)

        order = 10
        coef_list = []
            
        for n in range(-order, order+1) :

            real_coef = quad(lambda t : np.real(func(t,self.time_table, self.x_table, 1j*self.y_table) * np.exp(-n * 1j * t)), 0, tau, limit=100, full_output=1)[0]/tau
            imag_coef = quad(lambda t : np.imag(func(t,self.time_table, self.x_table, 1j*self.y_table) * np.exp(-n * 1j *t)), 0, tau, limit=100, full_output=1)[0]/tau

            self.real_coef = real_coef
            self.imag_coef = imag_coef

                
            coef_list.append([real_coef, imag_coef])

        return np.array(coef_list)


    def DFT(self, t, coef_list, order=10):
        
        kernel = np.array([np.exp(-n*1j*t) for n in range(-order, order+1)])
        series = np.sum((coef_list[:,0]+1j*coef_list[:,1]) * kernel[:])

        self.kernel = kernel
        self.series = series

        return np.real(series), np.imag(series)
