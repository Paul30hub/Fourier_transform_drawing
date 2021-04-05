from __future__ import division
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import math
import sympy
sympy.init_printing()

#Objectives : 
# - Find the Fourier series coefficients for a square wave
# - Interpret square wave as a weighted sum of sinusoidal building blocks

# Main functions

class Fourier_Series_Coeff : 

    def __init__(self, f, N, offset=0, period=2*np.pi) :

        self.N = N 
        self.period = period
        self.offset = offset 
        self.c = np.zeros(2*N+1, dtype=complex)

        for n in range(-N,N+1):

            self.c[n] = complex(integrate.quad(lambda x : (f(x)*np.exp(-2.0j*np.pi*n*x/period)).real,\
                offset, offset + period)[0], \
            integrate.quad(lambda x : (f(x)*np.exp(-2.0j*np.pi*n*x/period)).imag, \
                offset, offset + period)[0])/period
            self._v_sn = np.vectorize(self._sn)


    # function to generate x+iy at given time t
    def _f(self,t, t_list, x_list, y_list):

        self.t = t
        self.t_list = t_list
        self.x_list = x_list
        self.y_list = y_list
        return np.interp(t, t_list, x_list + 1j*y_list)
    

    def _sn(self, x) : 
        ans = 0.0j

        for n in range(-self.N, self.N+1) : 
            ans += self.c[n] * np.exp(2.0j*np.pi*n*(x-self.offset)/self.period)
        return ans 


    def __call__(self, x) : 

        return self._v_sn(x) 


    def FourierSeriesApprox(self,xvals,yvals,nmax):

        self.xvals = xvals
        self.yvals = yvals
        self.nmax = nmax

        approx=np.zeros_like(yvals)
        T=(xvals[-1]-xvals[0])
        w=2*np.pi/T
        dt=xvals[1]-xvals[0]
        approx=approx+1/T*(np.sum(yvals)*dt)

        for t in range(len(xvals)):
            for n in (np.arange(nmax)+1):
                an=2/T*np.sum(np.cos(w*n*xvals)*yvals)*dt
                bn=2/T*np.sum(np.sin(w*n*xvals)*yvals)*dt
                approx[t]=approx[t]+an*np.cos(w*n*xvals[t])+bn*np.sin(w*n*xvals[t])

        return approx