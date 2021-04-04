# Find the fourier coefficient to approximately locate given points
# f(t)=(x(t),y(t))
# Fourer coefficicent formula is : Cn = (1/2pi)int_{0}^{2pi}f(t)exp(-int)dt
# Where Cn = coefficient calculated wich is in form "x+iy" ou "r*exp(-int)" where r is radius
# of circle and t gives the position of point in circumference of circle f(t) return x,y points at time t.
# Coefficient will be in sequence like : ...c_{-2},c_{-1},c_{0},c_{1},c_{2}...
# More and more coefficients means better result.