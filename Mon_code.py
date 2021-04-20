import cv2 # for finding contours to extract points of figure
import matplotlib.pyplot as plt # for plotting and creating figures
import numpy as np # for easy and fast number calculation
from math import tau # tau is constant number = 2*PI
from scipy.integrate import quad_vec, quad # for calculating definite integral 
from tqdm import tqdm # for progress bar
import matplotlib.animation as animation # for compiling animation and exporting video 
#%%

### Partie Kenjy

def create_close_loop(image_name):
    # read image to array, then get image border with contour
    # reading the image and convert to greyscale mode
    # ensure that you use image with black image with white background
    img = cv2.imread(image_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get Contour Path and create lookup-table
    # find the contours in the image
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0) # making pure black and white image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # finding available contours i.e closed loop objects
    contours = np.array(contours[1]) # contour at index 1 is the one we are looking for

    # split the co-ordinate points of the contour
    # we reshape this to make it 1D array
    x_list, y_list = contours[:, :, 0].reshape(-1,), -contours[:, :, 1].reshape(-1,)
    # time data from 0 to 2*PI as x,y is the function of time.
    t_list = np.linspace(0, tau, len(x_list)) # now we can relate f(t) -> x,y

    # center the contour to origin
    x_list = x_list - np.mean(x_list)
    y_list = y_list - np.mean(y_list)

    # visualize the contour
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_list, y_list)

    # later we will need these data to fix the size of figure
    xlim_data = plt.xlim() 
    ylim_data = plt.ylim()

    return t_list, x_list, y_list

#%%
t_list, x_list, y_list = create_close_loop("star.jpeg")
print(t_list, x_list, y_list)
#%%

##### Partie Chloe 

# function to generate x+iy at given time t
def f(t, t_list, x_list, y_list):
    return np.interp(t, t_list, x_list + 1j*y_list)

#%%
def coef_list(t_list, x_list, y_list, order=10):
    coef_list = []
    for n in range(-order, order+1):
        real_coef = quad(lambda t: np.real(f(t, t_list, x_list, y_list) * np.exp(-n*1j*t)), 0, tau, limit=100, full_output=1)[0]/tau
        imag_coef = quad(lambda t: np.imag(f(t, t_list, x_list, y_list) * np.exp(-n*1j*t)), 0, tau, limit=100, full_output=1)[0]/tau
        coef_list.append([real_coef, imag_coef])
    return np.array(coef_list)

#%%

order = 10
cf = coef_list(t_list, x_list, y_list, order)
print(cf)

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

            plt.pause(1e-10)

if __name__ == '__main__':
    '''
    Example :
    with 3 circles and 2 repetitions
    '''
    cir = len(cf)
    cycles = 1
    fcoef = np.empty(len(cf), dtype=complex)
    for i in range(len(cf)):
        fcoef[i] = np.complex(cf[i][0],cf[i][1])
    fs = FS(cir,cycles,fcoef)
    fs.PlotFS()