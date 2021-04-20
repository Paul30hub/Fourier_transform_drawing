# import packages
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from numpy import interp
from math import tau
from scipy.integrate import quad

class ImageReader:
    """
    Class image
    """
    def __init__(self, img):
        self.img = Image.open(img)
    
    def get_tour(self, level=[200]):
        #prepare plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_aspect('equal', 'datalim')
        ax[1].set_aspect('equal', 'datalim')
        ax[0].set_title('Before Centered')
        ax[1].set_title('After Centered')

        # read image to array, then get image border with contour
        im = np.array(self.img.convert('L'))
        contour_plot = ax[0].contour(im, levels=level, colors='black', origin='image')

        # Get Contour Path and create lookup-table
        contour_path = contour_plot.collections[0].get_paths()[0]
        x_table, y_table = contour_path.vertices[:, 0], contour_path.vertices[:, 1]
        time_table = np.linspace(0, tau, len(x_table))

        # Simple method to center the image
        x_table = x_table - min(x_table)
        y_table = y_table - min(y_table)
        x_table = x_table - max(x_table) / 2
        y_table = y_table - max(y_table) / 2

        self.x_table = x_table
        self.y_table = y_table
        self.time_table = time_table

        # Visualization
        ax[1].plot(self.x_table , self.y_table, 'k-')
        
        return self.time_table, self.x_table, self.y_table

#test function imageReader
#image = ImageReader("velo.jpeg")
#image.get_tour()