# My project

We divided the work into four between BURGAT Paul, KOAN Kenjy, GUILLAUMONT Pierre and me.

In this part, I will present my part of the work to you.

My part is to find the fourier coefficients to approximately locate given points : 
I will create a class Fourier_Series_Coeff for our package that provides functions for calculate the Fourier approximations of the extracted path and use coefficients from this approximation to determine the phase and amplitude of the circle needed for the final visualization.

__________________________________________________________________________________________________________________________________________________________________________________________________________________________________

First, I would like to apply the formula for Fourier coefficients with respect to several simple functions.
Then in a second step, I would like to apply some more concrete examples, ie on an "image processing" finally to be able to redraw an image of .png type that we downloaded.


## Fourier Series :


<img width="316" alt="Horse" src="https://user-images.githubusercontent.com/81428023/114719045-ede5fe80-9d36-11eb-9ed3-0ca7185e9f2c.png">


We take the image above in order to calculate the Fourier approximation.

Our class of functions allows us to do this work on the images which will be processed by the class made by KOAN Kenjy. This is the logical continuation of his work.

When we study Fourier transformations, we have several elements to take into account :

- Function to bz transformed (as Python function object)
- Tuple of range at which to evaluate our function 
- A number of coefficients to calculate
- Our period : f different than full length of function
- Number of points : Number of points at which to evalute function
- Number of circles : This is needed to calculate proper offsets 
