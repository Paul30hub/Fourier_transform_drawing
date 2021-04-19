import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import ConnectionPatch
import cmath

#%% Class for Fourier series

class FS():

    def __init__(self, Circles, Cycles, fcoef): #number of circles and number of cycles and fourier coefficients 
        '''
        Number of circles and cycles and Fourier coefficients
        '''
        self.Circles = Circles
        self.Cycles = Cycles
        self.fcoef = fcoef

    def Xcenter(self, n, theta): # X coordinates of the center of the circle
        
        '''
           X coordinate of n th circle
        '''
        Ans = 0

        if n>0:
            for i in range (0, n):
                # Ans -=np.cos( (i+1)* theta)/ ((i+1)* np.pi) 
                Ans -= np.cos( (i+1) * theta + cmath.polar(self.fcoef[i])[1] ) * abs(self.fcoef[i])

        return Ans

    def Ycenter(self, n, theta): # Y coordinates of the center of the circle
        '''
           Y coordinate of n th circle
        '''
        Ans = 0
        if n > 0:
            for i in range(0, n):
                Ans -= np.sin( (i+1) * theta + cmath.polar(self.fcoef[i])[1] ) * abs(self.fcoef[i])

        return Ans

    def PlotFS(self): # representation of Fourier serie
        '''
            Ploting Fourier serie 
        '''
        time = np.linspace(0, self.Cycles, self.Cycles* 200)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(80, 60))
        fig.suptitle('Fourier Series', fontsize = 45, fontweight = 'bold') 
        
        color = cm.rainbow( np.linspace(0, 1, self.Circles) )

        for t in time:

            thta = 2 * np.pi * t

            axs[0].clear()

            if (t > 0):
                con.remove()
            
            for i, c in zip(range(0, self.Circles ), color): #Animation of circles left part
                xc = self.Xcenter(i, thta)
                yc = self.Ycenter(i, thta)
                R  = abs(self.fcoef[i])   

                crl = plt.Circle((xc, yc), R, color = c, alpha = 0.5, linewidth = 2)
                axs[0].add_artist(crl)

                if (i > 0):
                    axs[0].plot([xco, xc], [yco, yc], color = 'b', linewidth = 2)

                xco = xc
                yco = yc
                
            xlim_plot = sum(np.absolute(np.real(self.fcoef)))
            ylim_plot = sum(np.absolute(np.imag(self.fcoef)))
            axs[0].axis('square')
            axs[0].set_xlim([ -xlim_plot * 2, xlim_plot * 2])
            axs[0].set_ylim([ -ylim_plot * 2, ylim_plot * 2])

            if (t > 0): #Curve drawn on the right side
                axs[1].plot(xco, ycirc,'.', color = 'm', linewidth = 1)

            to = t
            ycirc = yc
            
            axs[1].axis('square')
            axs[1].set_xlim([ -xlim_plot * 2, xlim_plot * 2])
            axs[1].set_ylim([ -ylim_plot * 2, ylim_plot * 2])

            # Creation of a red line between the 2 plots

            con = ConnectionPatch( xyA = (xc, yc), xyB = (xc, yc),
                                   coordsA = 'data', coordsB = 'data',
                                   axesA= axs[1], axesB= axs[0],
                                   color = 'red')
            axs[1].add_artist(con)

            plt.pause(1e-14)

if __name__ == '__main__':
    '''
    Example :
    with 3 circles and 2 repetitions

    '''
    cir = 3 
    cycles = 2 
    fcoef = [2 + 1j, 1+1j, 1-1j] 
    fs = FS(cir,cycles,fcoef)
    fs.PlotFS()
