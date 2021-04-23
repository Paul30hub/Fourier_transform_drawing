# import packages
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from numpy import interp
from math import tau
from scipy.integrate import quad
import requests
from io import BytesIO

class ImageReader:
    """
    Class image
    """
    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.img = Image.open(BytesIO(response.content))
        self.im = self.img.convert("L")
        
        
        
    
    def get_tour(self, level= [200]):

        contour_plot = plt.contour(self.im,levels = level,origin='image')
        contour_path = contour_plot.collections[0].get_paths()[0]
        x_table, y_table = contour_path.vertices[:,0], contour_path.vertices[:,1]

        #Center X and Y
        x_table = x_table - min(x_table)
        y_table = y_table - min(y_table)
        x_table = x_table - max(x_table)/2
        y_table = y_table - max(y_table)/2

        #Visualize
        plt.plot(x_table, y_table)
        #For the period of time 0-2*pi
        time_table = np.linspace(0,tau,len(x_table))

        self.x_table = x_table
        self.y_table = y_table
        self.time_table = time_table
        
        return self.time_table, self.x_table, self.y_table