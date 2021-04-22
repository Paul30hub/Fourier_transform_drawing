import matplotlib.pyplot as plt # for plotting and creating figures
import numpy as np # for easy and fast number calculation
from math import tau

import Package4
from Package4 import ImageReader
from Package4 import Fourier

image = ImageReader("Heart.png")
time_table, x_table, y_table = image.get_tour()

#print(time_table, x_table, y_table)
order = 50

cf = Fourier(time_table, x_table, y_table,order)
fouriercoeff = cf.coef_list(time_table, x_table, y_table,order)

print(fouriercoeff)