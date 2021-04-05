# My project

We divided the work into four between BURGAT Paul, KOAN Kenjy, GUILLAUMONT Pierre and me.

In this part, I will present my part of the work to you.

My part is to find the fourier coefficients to approximately locate given points : 
I will create a class for our package that provides functions for calculate the Fourier approximations of the extracted path and use coefficients from this approximation to determine the phase and amplitude of the circle needed for the final visualization.

__________________________________________________________________________________________________________________________________________________________________________________________________________________________________

First, I would like to apply the formula for Fourier coefficients with respect to several simple functions.
Then in a second step, I would like to apply some more concrete examples, ie on an "image processing" finally to be able to redraw an image of .png type that we downloaded.


## Fourier Series :

The idea of the Fourier series is as follows. 
Given a function f which is 2π-periodic, can it be written as a sum of elementary 2π-periodic functions ?  

So either f a function 2π-periodic, integrable on [0,2π], continues in pieces. We call exponential Fourier coefficients of f, the complex numbers defined by :

<img width="285" alt="Cn" src="https://user-images.githubusercontent.com/81428023/113521178-0ec08e00-9598-11eb-9e72-04083d4f7ef7.png">

The trigonometric Fourier coefficients are defined by:

<img width="263" alt="coeff" src="https://user-images.githubusercontent.com/81428023/113521199-2d268980-9598-11eb-82eb-fc55e6215ed1.png">

The Fourier series of f is then defined by:

<img width="298" alt="serie" src="https://user-images.githubusercontent.com/81428023/113521209-47f8fe00-9598-11eb-9236-6b7b1618f41a.png">

It can also be expressed with the exponential coefficients:

<img width="102" alt="exp" src="https://user-images.githubusercontent.com/81428023/113521218-56dfb080-9598-11eb-8efc-e9ff8e96eaab.png">

