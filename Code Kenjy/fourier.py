#%%
import cv2 # for finding contours to extract points of figure
import matplotlib.pyplot as plt # for plotting and creating figures
import numpy as np # for easy and fast number calculation
from math import tau # tau is constant number = 2*PI
from scipy.integrate import quad_vec # for calculating definite integral 
from tqdm import tqdm # for progress bar
import matplotlib.animation as animation # for compiling animation and exporting video 
#%%
# function to generate x+iy at given time t
def f(t, t_list, x_list, y_list):
    return np.interp(t, t_list, x_list + 1j*y_list)
#%%
# reading the image and convert to greyscale mode
# ensure that you use image with black image with white background
img = cv2.imread("paysage.jpeg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#%%
# find the contours in the image
ret, thresh = cv2.threshold(img_gray, 127, 255, 0) # making pure black and white image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # finding available contours i.e closed loop objects
contours = np.array(contours[1]) # contour at index 1 is the one we are looking for
#%%
# split the co-ordinate points of the contour
# we reshape this to make it 1D array
x_list, y_list = contours[:, :, 0].reshape(-1,), -contours[:, :, 1].reshape(-1,)
#%%
# center the contour to origin
x_list = x_list - np.mean(x_list)
y_list = y_list - np.mean(y_list)
#%%
# visualize the contour
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_list, y_list)
#%%
# later we will need these data to fix the size of figure
xlim_data = plt.xlim() 
ylim_data = plt.ylim()
#%%
plt.show()
# %%
