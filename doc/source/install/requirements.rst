*************
Sparse matrix
*************


What is this 
^^^^^^^^^^^^^

First of all, it must be said that a sparse matrix is ​​a matrix in which most of the elements are zero and only a few elements are different from zero.

In Python, these sparse matrices, based mainly on NumPy arrays, are efficiently implemented in the scipy.sparse submodule of the SciPy library which has been implemented according to the following idea: instead of storing all the values ​​in a matrix dense, it is easier to store non-zero values ​​in any format.

The best performance in terms of time and space is obtained when we store a sparse array with the **scipy.sparse** submodule.

The advantage of sparse matrices is to be able to handle matrices, potentially enormous, but which have a very small number of non-zero coefficients compared to the numbers of inputs of the matrix, for example less than 1%.


To check if a matrix is ​​sparse or not, we will use the library

.. code-block:: Python

   from scipy.sparse import isspmatrix


isspmatrix returns a boolean: TRUE or FALSE depending on whether the matrix is ​​sparse or not.


We are therefore going to check if our matrix of the fourier coefficients is sparse or not.

.. code-block:: Python

  isspmatrix(fouriercoeff)



This command returns the Boolean FALSE so in the end our matrix of Fourier coefficients is **not a sparse matrix**.


****************
Time efficiency
****************

============  =====================           
Functions     Execution time(s)       
============  =====================
get_tour       0.67347s
coef_list      14.4338s
visualize      26.20057s
============  =====================  

