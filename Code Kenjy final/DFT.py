#%%
import cv2 # for finding contours to extract points of figure
import matplotlib.pyplot as plt # for plotting and creating figures
import numpy as np # for easy and fast number calculation
from math import tau # tau is constant number = 2*PI
from scipy.integrate import quad_vec, quad # for calculating definite integral 
from tqdm import tqdm # for progress bar
import matplotlib.animation as animation # for compiling animation and exporting video 
#%%

### Partie Kenjy

def create_close_loop(image_name):
    # read image to array, then get image border with contour
    # reading the image and convert to greyscale mode
    # ensure that you use image with black image with white background
    img = cv2.imread(image_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get Contour Path and create lookup-table
    # find the contours in the image
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0) # making pure black and white image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # finding available contours i.e closed loop objects
    contours = np.array(contours[1]) # contour at index 1 is the one we are looking for

    # split the co-ordinate points of the contour
    # we reshape this to make it 1D array
    x_list, y_list = contours[:, :, 0].reshape(-1,), -contours[:, :, 1].reshape(-1,)
    # time data from 0 to 2*PI as x,y is the function of time.
    t_list = np.linspace(0, tau, len(x_list)) # now we can relate f(t) -> x,y

    # center the contour to origin
    x_list = x_list - np.mean(x_list)
    y_list = y_list - np.mean(y_list)

    # visualize the contour
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_list, y_list)

    # later we will need these data to fix the size of figure
    xlim_data = plt.xlim() 
    ylim_data = plt.ylim()

    return t_list, x_list, y_list

#%%
t_list, x_list, y_list = create_close_loop("star.jpeg")
print(t_list, x_list, y_list)
#%%

##### Partie Chloe 

# function to generate x+iy at given time t
def f(t, t_list, x_list, y_list):
    return np.interp(t, t_list, x_list + 1j*y_list)

#%%
def coef_list(t_list, x_list, y_list, order=10):
    coef_list = []
    for n in range(-order, order+1):
        real_coef = quad(lambda t: np.real(f(t, t_list, x_list, y_list) * np.exp(-n*1j*t)), 0, tau, limit=100, full_output=1)[0]/tau
        imag_coef = quad(lambda t: np.imag(f(t, t_list, x_list, y_list) * np.exp(-n*1j*t)), 0, tau, limit=100, full_output=1)[0]/tau
        coef_list.append([real_coef, imag_coef])
    return np.array(coef_list)

#%%

order = 50
cf = coef_list(t_list, x_list, y_list, order)
print(cf)
#%%
def DFT(t, coef_list, order=10):
    kernel = np.array([np.exp(-n*1j*t) for n in range(-order, order+1)])
    series = np.sum( (coef_list[:,0]+1j*coef_list[:,1]) * kernel[:])
    return np.real(series), np.imag(series)

#%%
space  = np.linspace(0,tau,300)
xdft = [DFT(t,cf,order)[0] for t in space]
ydft = [DFT(t,cf,order)[1] for t in space]

#%%
fig,ax = plt.subplots(figsize = (5,5))
ax.plot(xdft,ydft,'r--')
ax.plot(x_list,y_list,'k-')
ax.set_aspect('equal','datalim')
xmin, xmax = plt.xlim()
ymin,ymax = plt.ylim()


#%%

#### Partie Pierre Paul

def visualize(x_DFT, y_DFT, coef, order, space, fig_lim):
    fig, ax = plt.subplots()
    lim = max(fig_lim)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_aspect('equal')

    # Initialize
    line = plt.plot([], [], 'k-', linewidth=2)[0]
    radius = [plt.plot([], [], 'r-', linewidth=0.5, marker='o', markersize=1)[0] for _ in range(2 * order + 1)]
    circles = [plt.plot([], [], 'r-', linewidth=0.5)[0] for _ in range(2 * order + 1)]

    def update_c(c, t):
        new_c = []
        for i, j in enumerate(range(-order, order + 1)):
            dtheta = -j * t
            ct, st = np.cos(dtheta), np.sin(dtheta)
            v = [ct * c[i][0] - st * c[i][1], st * c[i][0] + ct * c[i][1]]
            new_c.append(v)
        return np.array(new_c)

    def sort_velocity(order):
        idx = []
        for i in range(1,order+1):
            idx.extend([order+i, order-i]) 
        return idx    
    
    def animate(i):
        # animate lines
        line.set_data(x_DFT[:i], y_DFT[:i])
        # animate circles
        r = [np.linalg.norm(coef[j]) for j in range(len(coef))]
        pos = coef[order]
        c = update_c(coef, i / len(space) * tau)
        idx = sort_velocity(order)
        for j, rad, circle in zip(idx,radius,circles):
            new_pos = pos + c[j]
            rad.set_data([pos[0], new_pos[0]], [pos[1], new_pos[1]])
            theta = np.linspace(0, tau, 50)
            x, y = r[j] * np.cos(theta) + pos[0], r[j] * np.sin(theta) + pos[1]
            circle.set_data(x, y)
            pos = new_pos
                
    # Animation
    ani = animation.FuncAnimation(fig, animate, frames=len(space), interval=5)
    return ani
# %%
anim = visualize(xdft, ydft, cf, order,space, [xmin, xmax, ymin, ymax])

#Change based on what writer you have
#HTML(anim.to_html5_video())
#anim.save('pi.gif',writer='ffmpeg')
anim.save('leaf.gif',writer='pillow')

#anim.save('epicycle.mp4')
#print("completed: epicycle.mp4")
# %%
