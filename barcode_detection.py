# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:09:38 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import io, color, measure
from scipy import signal
from scipy.signal import fftconvolve
from Blob import Blob
from time import time
from Blob import Blob
from math import floor, sqrt
import os
from common import * 
start_time = time()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Fonctions préliminaires ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

def D(X, Y, Z): return np.sqrt((X-Y)**2+4*Z**2)/(X+Y)

def barcode_detection(img,sigma_g,sigma_t,seuil,sigma_bruit=2,affichage=False):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Préparation de l'image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # %%
    img_code_barre = plt.imread(img)
    # ~~~~~~~~~~~~~~~~~~~~ transformation de rgb en ycrcb  ~~~~~~~~~~~~~~~~~~~
    img_code_barre_YCbCr = color.rgb2ycbcr(img_code_barre)
    Y_code_barre = img_code_barre_YCbCr[:, :, 0]
    Cb_code_barre = img_code_barre_YCbCr[:, :, 1]
    Cr_code_barre = img_code_barre_YCbCr[:, :, 2]
    Y_code_barre += np.random.randn(len(Y_code_barre),len(Y_code_barre[0]))*sigma_bruit
    # ~~~~~~~~~~~~~~~~~~~ Gradients ~~~~~~~~~~~~~~~~~~~
    gauss_x_prime = G_x_prime(sigma_g)
    gauss_y_prime = G_y_prime(sigma_g)
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
    D_res = D(Txx, Tyy, Txy)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seuillage ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    D_seuil = (D_res > seuil)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Labelisation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    D_labeled,num_labels=measure.label(D_seuil,return_num=True)
    blobs=measure.regionprops(D_labeled)
    print(f"{num_labels} objects detected in img")
    coords=[x.coords for x in blobs]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Extraction de l'axe ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # En utilisant la méthode vectorielle
    Blobs=[Blob(pixels=x.coords,imsize=[h,w]) for x in blobs]
    axis=[b.calc_axis_ray(6) for b in Blobs]
    
    # Methode des points extrêmes
    # Blobs=[Blob(pixels=x.coords,imsize=[h,w]) for x in blobs]
    # axis=[b.calc_axis_extr() for b in Blobs]
    #============================================================================================================
    # affichage basique des rayons obtenus
    plt.figure()
    plt.figure(figsize=(10, 5), dpi=300)
    plt.subplot(1, 2, 1)
    plt.imshow(img_code_barre)
    for blob in Blobs:
        # assert(blob.axis!=None)
        x=[u[0] for u in blob.axis]
        y=[u[1] for u in blob.axis]
        p=blob.barycentre
        plt.plot(p[1],p[0],"or",markersize=2.5)
        plt.plot(x,y,"+b",markersize=5)
    plt.title("img originale")
    plt.subplot(1, 2, 2)
    plt.imshow(D_labeled, cmap=cm.BrBG_r)
    for blob in Blobs:
        # assert(blob.axis!=None)
        x=[u[0] for u in blob.axis]
        y=[u[1] for u in blob.axis]
        p=blob.barycentre
        plt.plot(p[1],p[0],"or",markersize=2.5)
        plt.plot(x,y,"+b",markersize=5)
    plt.title("img labélisée")
    plt.colorbar()
    # aff=[b.__repr__() for b in Blobs]
    plt.show()
    if affichage:
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

        # plt.figure(6)
        # plt.subplot(1, 2, 1)
        # plt.imshow(img_code_barre)
        # plt.title("img origine")
        # plt.subplot(1, 2, 2)
        # plt.imshow(D_labeled, cmap=cm.BrBG_r)
        # for blob in Blobs:
        #     assert(blob.axis!=None)
        #     a,b=blob.axis
        #     p=blob.barycentre
        #     plt.plot(p[1],p[0],"or",markersize=2.5)
        #     plt.plot(a[0],a[1],"+b",markersize=5)
        #     plt.plot(b[0],b[1],"+b",markersize=5)
        # plt.title("img labelisee")

        # aff=[b.__repr__() for b in Blobs]
        
        plt.show()
        
    return Blobs

def barcode_detection_2(img,sigma_g,sigma_t,seuil,sigma_bruit=2,affichage=False):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Préparation de l'image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # %%
    img_code_barre = plt.imread(img)
    # ~~~~~~~~~~~~~~~~~~~~ transformation de rgb en ycrcb  ~~~~~~~~~~~~~~~~~~~
    img_code_barre_YCbCr = color.rgb2ycbcr(img_code_barre)
    Y_code_barre = img_code_barre_YCbCr[:, :, 0]
    Cb_code_barre = img_code_barre_YCbCr[:, :, 1]
    Cr_code_barre = img_code_barre_YCbCr[:, :, 2]
    Y_code_barre += np.random.randn(len(Y_code_barre),len(Y_code_barre[0]))*sigma_bruit
    # ~~~~~~~~~~~~~~~~~~~ Gradients ~~~~~~~~~~~~~~~~~~~
    gauss_x_prime = G_x_prime(sigma_g)
    gauss_y_prime = G_y_prime(sigma_g)
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
    D_res = D(Txx, Tyy, Txy)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seuillage ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    D_seuil = (D_res > seuil)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Labelisation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    D_labeled,num_labels=measure.label(D_seuil,return_num=True)
    blobs=measure.regionprops(D_labeled)
    print(f"{num_labels} object(s) detected in img")
    coords=[x.coords for x in blobs]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Extraction de l'axe ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # En utilisant la méthode vectorielle
    Blobs=[Blob(pixels=x.coords,imsize=[h,w]) for x in blobs]
    # axis=[b.calc_axis_ray(6) for b in Blobs]
    
    # Methode des points extrêmes
    # Blobs=[Blob(pixels=x.coords,imsize=[h,w]) for x in blobs]
    # axis=[b.calc_axis_extr() for b in Blobs]
    #============================================================================================================
    return Blobs
