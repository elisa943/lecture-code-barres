import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from math import ceil, floor, sqrt
from skimage import io, color, feature
from copy import deepcopy
from scipy import signal, ndimage
from scipy.interpolate import RectBivariateSpline
from skimage.morphology import closing, square
from time import time
from scipy.signal import fftconvolve
start_time=time()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%
def read(title): return plt.imread("img/"+title)


def to_ycbcr(img): return (color.rgb2ycbcr(img)/255).astype("double")
def to_rgb(img): return color.ycbcr2rgb(img*255)


def plot_channels(channels, titles=None, res_factor=1):
    n = len(channels)
    if titles == None:
        titles = [str(i) for i in range(n)]
    # subplot
    h, w = ceil(n/2), 2
    # print(n)
    print((h, w))
    plt.figure(figsize=(4*res_factor*w, 3*res_factor*h))
    for i in range(n):
        plt.subplot(h, w, i+1)
        plt.imshow(channels[i], cmap='gray')
        plt.title(titles[i])

    # plt.tight_layout()
    return


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PARAMETRES FILTRES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%
# Pour le bruit
sigma_bruit=2.1

# Pour le gradient
sigma_g = 1     

# Pour le tenseur
sigma_t = 15

"""
sigma canny:
    relativement faible pour trouver les vecteurs de transition correspondant aux barres
sigma T:
    relativement élevé pour trouer des clusters de vecteurs gradient
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%
# load image

img_code_barre = plt.imread('img/code_barre_prof.jpg')

print("test")
# ~~~~~~~~~~~~~~~~~~~~ transformation de rgb en ycrcb  ~~~~~~~~~~~~~~~~~~~
img_code_barre_YCbCr = color.rgb2ycbcr(img_code_barre)
Y_code_barre = img_code_barre_YCbCr[:, :, 0]
Cb_code_barre = img_code_barre_YCbCr[:, :, 1]
Cr_code_barre = img_code_barre_YCbCr[:, :, 2]
Y_code_barre+= np.random.randn(len(Y_code_barre),len(Y_code_barre[0]))*sigma_bruit
# ~~~~~~~~~~~~~~~~~~~ filtre ~~~~~~~~~~~~~~~~~~~
print("test")

def G_2D(sigma):
    x = range(floor(-3*sigma), floor(3*sigma+1))
    X, Y = np.meshgrid(x, x)
    return np.exp(-1/2*(X**2/(sigma**2)+(Y**2/sigma**2)))


def G_x_prime(sigma):  # derive d'une gaussienne
    P = range(floor(-3*sigma), floor(3*sigma+1))
    X, Y = np.meshgrid(P, P)
    return (-X/(2*np.pi*sigma**4)*np.exp(-(X**2+Y**2)/(2*sigma**2)))



def G_y_prime(sigma):  # derive d'une gaussienne
    P = range(floor(-3*sigma), floor(3*sigma+1))
    X, Y = np.meshgrid(P, P)
    return (-Y/(2*np.pi*sigma**4)*np.exp(-(X**2+Y**2)/(2*sigma**2)))


print("test")
gauss_x_prime = G_x_prime(sigma_g)
gauss_y_prime = G_y_prime(sigma_g)
# ~~~~~~~~~~~~~~~~~~~ gradient ~~~~~~~~~~~~~~~~~~~
h, w, c = np.shape(img_code_barre_YCbCr)
x = np.linspace(0, h, h)
y = np.linspace(0, w, w)
X, Y = np.meshgrid(y, x)
print("test")
# I_x = signal.convolve2d(Y_code_barre, gauss_x_prime,1mode='same', boundary='fill', fillvalue=0)
I_x = fftconvolve(Y_code_barre, gauss_x_prime, mode='same')
print("test")
I_y = fftconvolve(Y_code_barre, gauss_y_prime, mode='same')
# ~~~~~~~~~~~~~~~~~~~ Normalisation ~~~~~~~~~~~~~~~~~~~
# default_value = (0, 0)
# delta_I=  [[default_value for _ in range(w)] for _ in range(h)]
print("test")
# for i in range(h):
#     for j in range(w):
#         delta_I[i][j]=(I_x[i][j],I_y[i][j])

delta_I = np.stack((I_x, I_y), axis=-1)

N_delta_I = np.linalg.norm(delta_I, ord=2, axis=-1)
# N_delta_I = [[np.sqrt(x**2 + y**2) for x, y in ligne] for ligne in delta_I]

print("test")
N_I_x=np.divide(I_x,N_delta_I)
N_I_y=np.divide(I_y,N_delta_I)
# ~~~~~~~~~~~~~~~~~~~ tenseur de structure local ~~~~~~~~~~~~~~~~~~~
print("test")
gauss2D = G_2D(sigma_t)
print("test")

Txx = fftconvolve(N_I_x * N_I_x, gauss2D, mode='same')
Tyy = fftconvolve(N_I_y * N_I_y, gauss2D, mode='same')
Txy = fftconvolve(N_I_x * N_I_y, gauss2D, mode='same')  

print("test")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mesure de cohérence ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%
# méthode avec les valeurs propres possibles seulement si la matrice est crée
T = np.block([[Txx, Txy], [Txy, Tyy]])
# dt=np.linalg.det(T)
# U, S, Vt = np.linalg.svd(T)
D = lambda X,Y,Z: np.sqrt((X-Y)**2+4*Z**2)/(X+Y)
print("test")
D_res=D(Txx,Tyy,Txy)
print("test")
D_seuil=(D_res>0.7)
print("test")
#plot_channels([img_code_barre,D_seuil])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Closing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

D_seuil_closed = closing(D_seuil, square(3))
print("test")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Affichage ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# plot_channels([img_code_barre, D_res, D_seuil, D_seuil_closed], ["image originale",
#               "mesure de cohérence", "image seuilée", "Image après Fermeture"], res_factor=2)

# # Affichage 1: Image originale et canal Y
# plot_channels([img_code_barre, img_code_barre_YCbCr[:, :, 0]], 
#               titles=["img origine", "img canal Y"], 
#               res_factor=2)

# # Affichage 2: Gradients x et y
# plot_channels([I_x, I_y], 
#               titles=["grad x", "grad y"], 
#               res_factor=2)

# # Affichage 3: Gradients normalisés x et y
# plot_channels([N_I_x, N_I_y], 
#               titles=["normalisé grad x", "normalisé grad y"], 
#               res_factor=2)

# # Affichage 4: Tenseurs Txx, Tyy et Txy
# plot_channels([Txx, Tyy, Txy], 
#               titles=["Txx", "Tyy", "Txy"], 
#               res_factor=2)

# # Affichage 5: Mesure D et D seuil
# plot_channels([D_res, D_seuil], 
#               titles=["D", "D seuil"], 
#               res_factor=2)

# plt.show()


print("test")

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

plt.figure(3)
plt.subplot(1, 2, 1)
plt.imshow(N_I_x,cmap='gray')
plt.title("normalisé grad x")
plt.subplot(1, 2, 2)
plt.imshow(N_I_y,cmap='gray')
plt.title("normalisé grad y")

plt.figure(4)
plt.subplot(1, 3, 1)
plt.imshow(Txx,cmap='gray')
plt.title("Txx")
plt.subplot(1, 3, 2)
plt.imshow(Tyy,cmap='gray')
plt.title("Tyy")
plt.subplot(1, 3, 3)
plt.imshow(Txy,cmap='gray')
plt.title("Txy")

plt.figure(5)
plt.subplot(1, 2, 1)
plt.imshow(D_res,cmap='gray')
plt.title("D")
plt.subplot(1, 2, 2)
plt.imshow(D_seuil,cmap='gray')
plt.title("D seuil")


# plt.show()
print(f"{time()-start_time} s")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Lancer aléatoire ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%
def bornage(h,w,p):
    if p[0]<0:
        p[0]=0
    if p[0]>h:
        p[0]=h
    if p[1]<0:
        p[1]=0 
    if p[1]>w:
        p[1]=w
    return p

# ajouter des paramètres pour avoir un tirage autre que uniforme

def random_ray_center(h,w,length):
    # méthode: centre, angle, longueur
    angle=np.random.uniform(0,2*np.pi)
    r=length/2
    
    center=np.array([np.random.randint(0,h),np.random.randint(0,w)])
    offset=np.array([np.cos(angle),np.sin(angle)])*r
    x1=center+offset
    x2=center-offset
    return np.int32([bornage(h,w,x1),bornage(h,w,x2)])

def random_ray(h,w,length):
    # méthode: extrémité1, angle, longueur
    angle=np.random.uniform(0,2*np.pi)
    
    x1=np.array([np.random.randint(0,h),np.random.randint(0,w)])
    offset=np.array([np.cos(angle),np.sin(angle)])*length
    x2=x1+offset
    
    return np.int32([bornage(h,w,x1),bornage(h,w,x2)])

