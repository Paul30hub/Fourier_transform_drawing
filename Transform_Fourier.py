#Fourier transform

#Inspired by the video :https://youtu.be/r6sGWTCMz2k?t=1000

#Create a package that can reproduce a similar video. 
#A particular interest on the parameters (degrees of approximation), 
#Speed of the videos would be of interest. 
#Test and analysis on various images should be investigated. 
#As a simple case, one would start with the constraint that the input image can be a black and white svg file. 
#Then, adaptation to color cases and to general image format (like png, jpg, etc. ) could be added.

#Simple examples
#Visualization of the real and imaginary part of the transform

import numpy as np
import matplotlib.pyplot as plt

n = 20

#definition of a
a = np.zeros(n)
a[1] = 1

#visualization of a
#add on the right the left value for the periodicity
plt.subplot(311)
plt.plot( np.append(a, a[0]) )


#calculation of A
A = np.fft.fft(a)

#visualization of A
#add on the right the left value for the periodicity
B = np.append(A, A[0])
plt.subplot(312)
plt.plot(np.real(B))
plt.ylabel("partie reelle")

plt.subplot(313)
plt.plot(np.imag(B))
plt.ylabel("partie imaginaire")

plt.show()

#Fftfreq function

#numpy.fft.fftfreq returns the frequencies of the signal calculated in the DFT.

#The returned freq array contains the discrete frequencies in number of cycles per time step. 
#For example if the time step is in seconds, then the frequencies will be given in cycles / second.

#If the signal contains n time step and the time step is worth d:
#freq = [0, 1,…, n / 2-1, -n / 2,…, -1] / (d * n) if n is even
#freq = [0, 1,…, (n-1) / 2, - (n-1) / 2,…, -1] / (d * n) if n is odd

import numpy as np
import matplotlib.pyplot as plt

# signal definition
dt = 0.1
T1 = 2
T2 = 5
t = np.arange(0, T1*T2, dt)
signal = 2*np.cos(2*np.pi/T1*t) + np.sin(2*np.pi/T2*t)

# signal display
plt.subplot(211)
plt.plot(t,signal)

# calculation of the Fourier transform and frequencies
fourier = np.fft.fft(signal)
n = signal.size
freq = np.fft.fftfreq(n, d=dt)

# display of the Fourier transform
plt.subplot(212)
plt.plot(freq, fourier.real, label="real")
plt.plot(freq, fourier.imag, label="imag")
plt.legend()

plt.show()