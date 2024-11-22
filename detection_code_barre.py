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

#filtre
def G_2D(n,sigma):
    x=range(-n,n)
    X,Y=np.meshgrid(x,x)
    return np.exp(-1/2*(X**2/(sigma**2)+(Y**2/sigma**2)))

def G_x_prime(n,sigma): #derive d'une gaussienne
    P = range(-n,n)
    X, Y = np.meshgrid(P,P)
    return (-X/(2*np.pi*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2)))

def G_y_prime(n,sigma): #derive d'une gaussienne
    P = range(-n,n)
    X, Y = np.meshgrid(P,P)
    return (-Y/(2*np.pi*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2)))

gauss2D=G_2D(2,1)
gauss_x_prime=G_x_prime(10,1)
gauss_y_prime=G_y_prime(10,1)
#gradient
h,w,c=np.shape(img_code_barre_YCbCr)
x = np.linspace(0, h, h)
y = np.linspace(0, w, w)
X, Y = np.meshgrid(y, x)

I_x=signal.convolve2d(Y_code_barre,gauss_x_prime,mode='same', boundary='fill', fillvalue=0)
I_y=signal.convolve2d(Y_code_barre,gauss_y_prime,mode='same', boundary='fill', fillvalue=0)

norm_I_x=np.linalg.norm(I_x,ord=2)
norm_I_y=np.linalg.norm(I_y,ord=2)

In_x=I_x/norm_I_x
In_y=I_y/norm_I_y

#tenseur de structure local

Txx=signal.convolve2d(I_x*I_x,gauss2D,mode='same', boundary='fill', fillvalue=0)
Tyy=signal.convolve2d(I_y*I_y,gauss2D,mode='same', boundary='fill', fillvalue=0)
Txy=signal.convolve2d(I_x*I_y,gauss2D,mode='same', boundary='fill', fillvalue=0)

# Mesure de coherence

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
plt.title("I_x")
plt.subplot(1, 2, 2)
plt.imshow(I_y,cmap='gray')
plt.title("I_y")

plt.figure(3)
plt.subplot(1, 3, 1)
plt.imshow(Txx,cmap='gray')
plt.title("Txx")
plt.subplot(1, 3, 2)
plt.imshow(Tyy,cmap='gray')
plt.title("Tyy")
plt.subplot(1, 3, 3)
plt.imshow(Txy,cmap='gray')
plt.title("Txy")

plt.figure(4)
plt.subplot(1, 2, 1)
plt.imshow(In_x,cmap='gray')
plt.title("I_x normalisée")
plt.subplot(1, 2, 2)
plt.imshow(In_y,cmap='gray')
plt.title("I_y normalisée")



plt.show()
