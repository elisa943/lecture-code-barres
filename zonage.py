# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:22:54 2024

@author: Admin
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from math import ceil, floor, sqrt
from skimage import io, color, feature
import skimage
from copy import deepcopy
from scipy import signal, ndimage
from scipy.interpolate import RectBivariateSpline

#~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%%
read = lambda title:plt.imread("img/"+title)
to_ycbcr= lambda img:(color.rgb2ycbcr(img)/255).astype("double")
to_rgb=lambda img:color.ycbcr2rgb(img*255)

def plot_channels(channels,titles=None,res_factor=1):
    n=len(channels)
    if titles==None:
        titles=[str(i) for i in range(n)]
    # subplot
    h,w=ceil(n/2),2
    # print(n)
    print((h,w))
    plt.figure(figsize=(4*res_factor*w, 3*res_factor*h))
    for i in range(n):
        plt.subplot(h,w,i+1)
        plt.imshow(channels[i])
        plt.title(titles[i])
    
    plt.tight_layout()
    plt.show()
    return 

def compute_gradient(image):
    # Convert to grayscale if the image is in color
    if image.ndim == 3:
        image = color.rgb2gray(image)
    
    # Compute gradients using Sobel operator
    sobel_x = ndimage.sobel(image, axis=0)  # horizontal gradient
    sobel_y = ndimage.sobel(image, axis=1)  # vertical gradient
    
    # Compute the magnitude of the gradient
    gradient_magnitude = np.hypot(sobel_x, sobel_y)
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max()  # Normalize to [0, 1]
    
    return sobel_x, sobel_y, gradient_magnitude


G = lambda x,y,sigma:(-x/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))) #derive d'une gaussienne

#~~~~~~~~~~~~~~~~~~~~~~~ Processing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%%
b=read("barcode0.jpg")

# def canny(X,Y):
#     Gx = -X/(2*np.pi*sig**3)*np.exp(-(X**2+Y**2)/(2*sig**2))
#     Gx = -X/(2*np.pi*sig**4)*np.exp(-(X**2+Y**2)/(2*sig**2))
#     return


#transformation de rgb en ycrcb
I=skimage.color.rgb2ycbcr(b)
Y=I[:,:,0]
Cb=I[:,:,1]
Cr=I[:,:,2]

sh=np.shape(Y)

x,y=list(range(0,sh[0])),list(range(0,sh[1]))
X,Y=np.meshgrid(x,y)
test=G(X,Y,3)
Ix,Iy=np.gradient(Y,x,y)
sigma=9
G_s=np.array([[G(i,j,sigma) for i in range(sh[1])]for j in range(sh[0])])

np.linalg.eig()


plot_channels([Ix,Iy],res_factor=2)







