from Projet_Chloe.Fourier_Coeff import Fourier_Series_Coeff

@np.vectorize

def f(x) : 
    return np.floor(x)%2

#t = time
S = Fourier_Series_Coeff(f, 10, 0., 2.)
t = np.linspace(0,4,200)
plt.plot(t, f(t), 'rx', t, S(t).real, 'b-' )
plt.legend(['Original','Fourier'])
plt.title('FOURIER SERIES--EXAMPLE 1')
plt.show()

#Example 2

from scipy.interpolate import interp1d

S1 = np.array([np.exp(4.0j*np.pi*x / 5.) for x in range(6)])
#print(S1)

_t = np.linspace(0., 2*np.pi, num = len(S1))
G = interp1d(_t, S1)

#Create a Fourier Series

FS = Fourier_Series_Coeff(G, 4)
t1 = np.linspace(0., 2*np.pi, 200)
FS.c

plt.plot(FS(t1).real, FS(t1).imag, 'b-', label='Fourier')
plt.plot(G(t1).real, G(t1).imag, 'r-', label='Interpolate')
plt. plot(S1.real, S1.imag, 'bx', label='Original')
plt.legend()
plt.title('FOURIER SERIES STAR--EXAMPLE 2')
plt.show()

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

#Example 3

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

#Example 4 : sawtooth = Return a periodic sawtooth or triangle waveform


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
plt.show() #Example 1 

@np.vectorize

def f(x) : 
    return np.floor(x)%2

#t = time
S = Fourier_Series_Coeff(f, 10, 0., 2.)
t = np.linspace(0,4,200)
plt.plot(t, f(t), 'rx', t, S(t).real, 'b-' )
plt.legend(['Original','Fourier'])
plt.title('FOURIER SERIES--EXAMPLE 1')
plt.show()

#Example 2

from scipy.interpolate import interp1d

S1 = np.array([np.exp(4.0j*np.pi*x / 5.) for x in range(6)])
#print(S1)

_t = np.linspace(0., 2*np.pi, num = len(S1))
G = interp1d(_t, S1)

#Create a Fourier Series

FS = Fourier_Series_Coeff(G, 4)
t1 = np.linspace(0., 2*np.pi, 200)
FS.c

plt.plot(FS(t1).real, FS(t1).imag, 'b-', label='Fourier')
plt.plot(G(t1).real, G(t1).imag, 'r-', label='Interpolate')
plt. plot(S1.real, S1.imag, 'bx', label='Original')
plt.legend()
plt.title('FOURIER SERIES STAR--EXAMPLE 2')
plt.show()

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

#Example 3

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

#Example 4 : sawtooth = Return a periodic sawtooth or triangle waveform


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