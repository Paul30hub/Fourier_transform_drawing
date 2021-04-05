# -*- coding: utf-8 -*-
# Find the fourier coefficient to approximately locate given points
# f(t)=(x(t),y(t))
# Fourer coefficicent formula is : Cn = (1/2pi)int_{0}^{2pi}f(t)exp(-int)dt, t = time
# Where Cn = coefficient calculated wich is in form "x+iy" ou "r*exp(-int)" where r is radius
# of circle and t gives the position of point in circumference of circle f(t) return x,y points at time t.
# Coefficient will be in sequence like for n circles : ...c_{-2},c_{-1},c_{0},c_{1},c_{2}...
# More and more coefficients means better result.
#So we will define order = 500  wich goes from c_{-order} to c_{order}:-500 to 500

order = 50 

#t_list, x_list, y_list, t=time 
import cv2 # for finding contours to extract points of figure
import matplotlib.pyplot as plt # for plotting and creating figures
import numpy as np # for easy and fast number calculation
from math import tau # tau is constant number = 2*PI
from scipy.integrate import quad_vec # for calculating definite integral
from scipy.signal import sawtooth
import math
#from math import* # import all function from math

#FIRST EXAMPLE

X = np.arange(-np.pi, np.pi, 0.001) # X axis has been chosen from -pi to pi, value of 
# 1 smallest square along X axis is 0.001

#Define f(X)=sqrt(X) any function
Y = np.sqrt(X)

#Fourier
fc = lambda X : np.sqrt(X) * np.cos(i*X) # i : dummy index
fs = lambda X : np.sqrt(X) * np.sin(i*X)


#Coeff a0, an, bn 
An = []
Bn = []
sum = 0 

for i in range(order):
    an = quad_vec(fc, 0, 2*np.pi)[0]
    An.append(an) # putting value in array An 

for i in range(order):
    bn = quad_vec(fs, 0, 2*np.pi)[0]
    Bn.append(bn) # putting value in array An 


for i in range(order):
    if i == 0:
        sum = sum + An[i]/2
    else :
        sum = sum + (An[i]*np.cos(i*X) + Bn[i]*np.sin(i*X))

plt.plot(X,sum,'b')
plt.plot(X,Y,'r')
plt.title("FOURIER SERIES")
plt.show()  

#2nd EXAMPLE : sawtooth = Return a periodic sawtooth or triangle waveform


X = np.arange(-np.pi, np.pi, 0.001)

#Define a sawtooth function
Y = sawtooth(X)

#Fourier
fc = lambda X : sawtooth(X) * np.cos(i*X) # i : dummy index
fs = lambda X : sawtooth(X) * np.sin(i*X)

#Coeff a0, an, bn 
An = []
Bn = []
sum = 0 

for i in range(order):
    an = quad_vec(fc, 0, 2*np.pi)[0]
    An.append(an) # putting value in array An 

for i in range(order):
    bn = quad_vec(fs, 0, 2*np.pi)[0]
    Bn.append(bn) # putting value in array An 


for i in range(order):
    if i == 0:
        sum = sum + An[i]/2
    else :
        sum = sum + (An[i]*np.cos(i*X) + Bn[i]*np.sin(i*X))

print("sum =", sum)
plt.plot(X,sum,'b')
plt.plot(X,Y,'r')
plt.title("FOURIER SERIES")
plt.show() 


#3 Example : more difficult : Nepal Map 

import PIL.Image
import PIL.ImageTk
import cv2 # for finding contours to extract points of figure
from cv2 import cv2
import matplotlib.pyplot as plt # for plotting and creating figures
import numpy as np # for easy and fast number calculation
from math import tau # tau is constant number = 2*PI
from scipy.integrate import quad_vec # for calculating definite integral 
from tqdm import tqdm # for progress bar
import matplotlib.animation as animation # for compiling animation and exporting video

# function to generate x+iy at given time t
def f(t, t_list, x_list, y_list):
    return np.interp(t, t_list, x_list + 1j*y_list)

#1-reading the image and convert to greyscale mode
# ensure that you use image with black image with white background
img = PIL.Image.open("/Users/coco/Desktop/Projet-Dev-Log/nepal.png")
print(img.size)
img_gray = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
