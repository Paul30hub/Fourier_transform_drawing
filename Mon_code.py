import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import ConnectionPatch

#%% Class pour les séries de Fourier

class FS():

    def __init__(self, Circles, Cycles, fcoef): #number of circles and number of cycles and fourier coef 

        self.Circles = Circles
        self.Cycles = Cycles
        self.fcoef = fcoef

    def Xcenter(self, n, theta): # Coordonnée X du centre du cercle
        
        '''
           X coordinate of n th circle
        '''
        Ans = 0

        if n>0:
            for i in range (0, n):
                # Ans -=np.cos( (i+1)* theta)/ ((i+1)* np.pi) 
                Ans -= np.cos( (i+1)* theta) * fcoef[i] 

        return Ans

    def Ycenter(self, n, theta): # coordonnée Y du centre du cercle
        '''
           Y coordinate of n th circle
        '''
        Ans = 0
        if n > 0:
            for i in range(0, n):
                # Ans -=np.sin( (i+1)* theta)/ ((i+1)* np.pi) 
                Ans -=np.sin( (i+1)* theta) * self.fcoef[i]

        return Ans

    def Rds(self, n): #Rayon du cercle
        '''
           Radius of n th circle
        '''

        return 1/((n+1)* np.pi)# radius of circle but not  radius 

    def PlotFS(self): #Représentation des séries de Fourier

        time = np.linspace(0, self.Cycles, self.Cycles* 250)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(80, 60))
        fig.suptitle('Fourier Series', fontsize = 45, fontweight = 'bold') 
        
        color = cm.rainbow( np.linspace(0, 1, self.Circles) )

        for t in time:

            thta = 2 * np.pi * t

            axs[0].clear()

            if (t > 0):
                con.remove()
            
            for i, c in zip(range(0, self.Circles ), color): #Premier plot
                xc = self.Xcenter(i, thta)
                yc = self.Ycenter(i, thta)
                R  = self.fcoef[i]  # self.Rds(i)###################### 

                crl = plt.Circle((xc, yc), R, color = c, alpha = 0.5, linewidth = 2)
                axs[0].add_artist(crl)

                if (i > 0):
                    axs[0].plot([xco, xc], [yco, yc], color = 'b', linewidth = 2)

                xco = xc
                yco = yc
            xylim_plot = sum(self.fcoef)
            axs[0].axis('square')
            axs[0].set_xlim([ -xylim_plot, xylim_plot])
            axs[0].set_ylim([ -xylim_plot, xylim_plot])

            if (t > 0): # Deuxième plot
                axs[1].plot(xco, ycirc,'.', color = 'm', linewidth = 1)

            to = t
            ycirc = yc
            
            axs[1].axis('square')
            axs[1].set_xlim([ -xylim_plot, xylim_plot ])
            axs[1].set_ylim([ -xylim_plot, xylim_plot])

            # Création d'une ligne rouge entre les 2 plots

            con = ConnectionPatch( xyA = (xc, yc), xyB = (xc, yc),
                                   coordsA = 'data', coordsB = 'data',
                                   axesA= axs[1], axesB= axs[0],
                                   color = 'red')
            axs[1].add_artist(con)

            plt.pause(1e-12)

if __name__ == '__main__':
    cir = int(input('number of circle(s) : '))
    cycles = int(input('number of cycles : '))
    fcoef =[] 
    for i in range(0, cir):
        fcoef.append(float(input('Fourier coefficient number {} : '.format(i + 1))))
    fs = FS(cir,cycles,fcoef)
    fs.PlotFS()