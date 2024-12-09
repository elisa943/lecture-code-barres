# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:24:36 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from math import ceil, floor, sqrt
from skimage import io, color, feature, measure
from copy import deepcopy
from scipy import signal, ndimage
from scipy.interpolate import RectBivariateSpline
from skimage.morphology import closing, square
from time import time
from scipy.signal import fftconvolve
# from numba import jit
start_time = time()
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

def bornage(h, w, p): # à voir si une accélération est possible
    if p[0] < 0:
        p[0] = 0
    if p[0] > h:
        p[0] = h
    if p[1] < 0:
        p[1] = 0
    if p[1] > w:
        p[1] = w
    return p

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PARAMETRES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%

# Pour le bruit, à regler à la main
sigma_bruit =1

# Pour le gradient, relativement faible pour trouver les vecteurs de transition correspondant aux barres
sigma_g = 1


# Pour le tenseur, relativement élevé pour trouer des clusters de vecteurs gradient
sigma_t = 15

seuil = 0.7
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Préparation de l'image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%
img_code_barre = plt.imread('img/code_barre_prof.jpg')
# ~~~~~~~~~~~~~~~~~~~~ transformation de rgb en ycrcb  ~~~~~~~~~~~~~~~~~~~
img_code_barre_YCbCr = color.rgb2ycbcr(img_code_barre)
Y_code_barre = img_code_barre_YCbCr[:, :, 0]
Cb_code_barre = img_code_barre_YCbCr[:, :, 1]
Cr_code_barre = img_code_barre_YCbCr[:, :, 2]
Y_code_barre += np.random.randn(len(Y_code_barre),len(Y_code_barre[0]))*sigma_bruit
# ~~~~~~~~~~~~~~~~~~~ Initialisation des filtres ~~~~~~~~~~~~~~~~~~~
# %%
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


gauss_x_prime = G_x_prime(sigma_g)
gauss_y_prime = G_y_prime(sigma_g)
# ~~~~~~~~~~~~~~~~~~~ Gradients ~~~~~~~~~~~~~~~~~~~

h, w, c = np.shape(img_code_barre_YCbCr)
x = np.linspace(0, h, h)
y = np.linspace(0, w, w)
X, Y = np.meshgrid(y, x)

I_x = fftconvolve(Y_code_barre, gauss_x_prime, mode='same')

I_y = fftconvolve(Y_code_barre, gauss_y_prime, mode='same')
# ~~~~~~~~~~~~~~~~~~~ Normalisation ~~~~~~~~~~~~~~~~~~~
delta_I = np.stack((I_x, I_y), axis=-1)

N_delta_I = np.linalg.norm(delta_I, ord=2, axis=-1)

N_I_x = I_x/N_delta_I
N_I_y = I_y/N_delta_I

# ~~~~~~~~~~~~~~~~~~~ tenseur de structure local T ~~~~~~~~~~~~~~~~~~~

gauss2D = G_2D(sigma_t)

Txx = fftconvolve(N_I_x * N_I_x, gauss2D, mode='same')
Tyy = fftconvolve(N_I_y * N_I_y, gauss2D, mode='same')
Txy = fftconvolve(N_I_x * N_I_y, gauss2D, mode='same')
T = np.block([[Txx, Txy], [Txy, Tyy]])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mesure de cohérence ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%
# méthode avec les valeurs propres possibles seulement si la matrice est carée
def D(X, Y, Z): return np.sqrt((X-Y)**2+4*Z**2)/(X+Y)


D_res = D(Txx, Tyy, Txy)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seuillage ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
D_seuil = (D_res > seuil)
# D_seuil_closed = closing(D_seuil, square(sigma_g*3)) # ne semble pas améliorer les performances significativement
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Labelisation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%

class Blob:
    def __init__(self, pixels=None,X=None,Y=None, barycentre=None, valeurs_propres=None, vecteurs_propres=None,area=None):
        self.pixels = pixels 
        self.X = X
        self.Y = Y 
        self.barycentre = barycentre 
        self.valeurs_propres = valeurs_propres 
        self.vecteurs_propres = vecteurs_propres 
        self.area=area
    def calc_XY(self):
        self.X,self.Y=self.pixels[:,0],self.pixels[:,1]
        return self.X,self.Y
    def calc_braycentre(self):
        self.barycentre=np.mean(self.pixels,axis=0)
        return self.barycentre
    def calc_vpropres(self):
        # self.calc_XY() if (None in [self.X,self.Y]) else print("X,Y déjà définis")
        self.valeurs_propres,self.vecteurs_propres=np.linalg.eig(np.cov(self.X,self.Y))
        return self.valeurs_propres,self.vecteurs_propres
    def calc_area(self):
        self.area=self.pixels.size # not the real area, but a good measure of how large it is
        return
    def calc_axis(self):
        
        return
    def calc_all(self):
        try:
            self.calc_XY()
            self.calc_braycentre()
            self.calc_vpropres()
            return True
        except Exception as e:
            print(e)
            return False
        
    def calc_axis(self):
        self.calc_all()
        p1 = floor(self.barycentre[1]+self.valeurs_propres[0]/2*self.vecteurs_propres[0][0]), floor(self.barycentre[0]+self.valeurs_propres[0]/2*self.vecteurs_propres[0][1])
        p2 = floor(self.barycentre[1]-self.valeurs_propres[0]/2*self.vecteurs_propres[0][0]), floor(self.barycentre[0]-self.valeurs_propres[0]/2*self.vecteurs_propres[0][1])
        return p1,p2
        
    # @property
    def __repr__(self):
        plt.figure()
        p1,p2 = self.calc_axis()
        plt.imshow(self.pixels)
        k=2
        h=floor((max(self.Y)+1-min(self.Y))*k)
        w=floor((max(self.X)+1-min(self.X))*k)
        I=np.zeros((w,h))
        # I[self.barycentre[0]-min(self.X)+floor(w/4),self.barycentre[1]-min(self.Y)+floor(h/4)]=3
        # affichage du Blob
        I[self.X-min(self.X)+floor(w/4),self.Y-min(self.Y)+floor(h/4)]=1
        
        # Affichage des points de l'axe
        a,b=p1[1]-min(self.X)+floor(w/4),p1[0]-min(self.Y)+floor(h/4)
        c,d=p2[1]-min(self.X)+floor(w/4),p2[0]-min(self.Y)+floor(h/4)
        
        I[p1[1]-min(self.X)+floor(w/4),p1[0]-min(self.Y)+floor(h/4)]=5
        I[p2[1]-min(self.X)+floor(w/4),p2[0]-min(self.Y)+floor(h/4)]=5
        # barycentre
        I[floor(w/2),floor(h/2)]=3
        plt.imshow(I,cmap=cm.hot)
        plt.colorbar()
        return str("Plot du Blob")
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Labelisation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

D_labeled,num_labels=measure.label(D_seuil,return_num=True)
blobs=measure.regionprops(D_labeled)
print(f"{num_labels} objects detected in img")
coords=[x.coords for x in blobs]

# en implémentant la classe:
Blobs=[Blob(pixels=x.coords) for x in blobs]
axis=[b.calc_axis() for b in Blobs]

#print(barycentres)
# print("valeur propres")
# print(valeurs_propres)
# print("vecteur propre")
# print(vecteurs_propres[0])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Affichage ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(img_code_barre)
plt.title("img origine")
plt.subplot(1, 2, 2)
plt.imshow(img_code_barre_YCbCr[:, :, 0], cmap='gray')
plt.title("img canal Y")

plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(I_x, cmap='gray')
plt.title("grad x")
plt.subplot(1, 2, 2)
plt.imshow(I_y, cmap='gray')
plt.title("grad y")

plt.figure(3)
plt.subplot(1, 2, 1)
plt.imshow(N_I_x, cmap='gray')
plt.title("normalisé grad x")
plt.subplot(1, 2, 2)
plt.imshow(N_I_y, cmap='gray')
plt.title("normalisé grad y")

plt.figure(4)
plt.subplot(1, 3, 1)
plt.imshow(Txx, cmap='gray')
plt.title("Txx")
plt.subplot(1, 3, 2)
plt.imshow(Tyy, cmap='gray')
plt.title("Tyy")
plt.subplot(1, 3, 3)
plt.imshow(Txy, cmap='gray')
plt.title("Txy")

plt.figure(5)
plt.subplot(1, 2, 1)
plt.imshow(D_res, cmap='gray')
plt.title("D")
plt.subplot(1, 2, 2)
plt.imshow(D_seuil, cmap='gray')
plt.title("D seuil")

plt.figure(6)
plt.subplot(1, 2, 1)
plt.imshow(img_code_barre)
plt.title("img origine")
plt.subplot(1, 2, 2)
plt.imshow(D_labeled, cmap=cm.BrBG_r)
for blob in Blobs:
    a,b=blob.calc_axis()
    p=blob.barycentre
    plt.plot(p[1],p[0],"or",markersize=2.5)
    plt.plot(a[0],a[1],"+b",markersize=5)
    plt.plot(b[0],b[1],"+b",markersize=5)
plt.title("img labelisee")

affichage=[b.__repr__() for b in Blobs]

plt.show() # rajoute plus d'une seconde à l'exécution, pourquoi?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Fonctions tracé de rayons ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%

def bornage2(h,w,ray):
    return np.array(bornage(h,w,ray[0]),bornage(h,w,ray[1]))

# ajouter des paramètres pour avoir un tirage gaussien (ou autre)

def ray_center(center,length,angle):
    """
    Parameters
    ----------
    center : np.array([x0,y0])
    length : float
    angle : float
    crée un rayon de centre center, de longueur length et d'angle alpha (par rapport à l'horizontale)
    """
    r = length/2
    offset = np.array([np.cos(angle), np.sin(angle)])*r
    x1 = center+offset
    x2 = center-offset
    return np.array([x1,x2])

def random_ray_center(h, w, length):
    # méthode: centre, angle, longueur
    angle = np.random.uniform(0, 2*np.pi)
    r = length/2
    center = np.array([np.random.randint(0, h), np.random.randint(0, w)])
    offset = np.array([np.cos(angle), np.sin(angle)])*r
    x1 = center+offset
    x2 = center-offset
    return np.int32([bornage(h, w, x1), bornage(h, w, x2)])


def random_ray(h, w, length):
    # méthode: extrémité1, angle, longueur
    angle = np.random.uniform(0, 2*np.pi)

    x1 = np.array([np.random.randint(0, h), np.random.randint(0, w)])
    offset = np.array([np.cos(angle), np.sin(angle)])*length
    x2 = x1+offset

    return np.int32([bornage(h, w, x1), bornage(h, w, x2)])


print(f"Code exécuté en {round(time()-start_time,3)} s")