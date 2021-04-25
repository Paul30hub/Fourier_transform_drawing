#%%
import matplotlib.pyplot as plt # for plotting and creating figures
import numpy as np # for easy and fast number calculation
from math import tau

import draw_circle_fourier
from draw_circle_fourier import ImageReader
from draw_circle_fourier import Fourier
from draw_circle_fourier import DrawAnimation

#Part Kenjy 

image = ImageReader("https://raw.githubusercontent.com/Paul30hub/Fourier_transform_drawing/main/PACKAGE/draw_circle_fourier/DATA/velo.jpeg")
time_table, x_table, y_table = image.get_tour()
print(time_table, x_table, y_table)

#Part Chloe

order = 20

cf = Fourier(time_table, x_table, y_table,order)
fouriercoeff = cf.coef_list(time_table, x_table, y_table, order)

print(fouriercoeff)

#Part Paul+Pierre

space = np.linspace(0, tau, 300)
x_DFT = [cf.DFT(t, fouriercoeff, order)[0] for t in space]
y_DFT = [cf.DFT(t, fouriercoeff, order)[1] for t in space]

fig, ax = plt.subplots(figsize = (5, 5))
ax.plot(x_DFT, y_DFT, 'r--')
ax.plot(x_table, y_table, 'k-')
ax.set_aspect('equal', 'datalim')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

b = DrawAnimation(x_DFT, y_DFT, fouriercoeff, order, space, [xmin, xmax, ymin, ymax])
anim = b.visualize(x_DFT, y_DFT, fouriercoeff, order, space, [xmin, xmax, ymin, ymax])

#Change based on what writer you have
#HTML(anim.to_html5_video())
#anim.save('pi.mp4',writer='ffmpeg')
anim.save('velo.gif', writer = 'pillow')
# %%
