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