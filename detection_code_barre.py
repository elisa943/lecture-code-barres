import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import signal
from matplotlib.colors import ListedColormap
from scipy.interpolate import RectBivariateSpline

# load image
img_code_barre=plt.imread('img/barcode0.jpg')

#transformation de rgb en ycrcb
img_code_barre_YCbCr=skimage.color.rgb2ycbcr(img_code_barre)
Y_code_barre=img_code_barre_YCbCr[:,:,0]
Cb_code_barre=img_code_barre_YCbCr[:,:,1]
Cr_code_barre=img_code_barre_YCbCr[:,:,2]

#gradient
h,w,c=np.shape(img_code_barre_YCbCr)
x = np.linspace(0, h, h)
y = np.linspace(0, w, w)
X, Y = np.meshgrid(y, x)

I_x,I_y=np.gradient(Y_code_barre,x,y)
#filtre
def G(x,y,sigma): #derive d'une gaussienne
    return (-x/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2)))

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(img_code_barre)
plt.title("img origine")
plt.subplot(1, 2, 2)
plt.imshow(img_code_barre_YCbCr[:,:,0],cmap='gray')
plt.title("img canal Y")

plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(I_x,cmap='gray')
plt.title("grad x")
plt.subplot(1, 2, 2)
plt.imshow(I_y,cmap='gray')
plt.title("grad y")



plt.show()
