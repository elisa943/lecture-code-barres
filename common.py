# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:58:05 2024

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt

def bornage(h, w, p): # à voir si une accélération est possible
    if p[0] < 0:
        p[0] = 0
    if p[0] > h:
        p[0] = h-1
    if p[1] < 0:
        p[1] = 0
    if p[1] > w:
        p[1] = w-1
    return p

def bornage2(h,w,ray):
    # unused for now
    ray=np.array(ray)
    return [bornage(h,w,ray[0]),bornage(h,w,ray[1])]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Fonctions tracé de rayons ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%
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
    offset = np.array([np.cos(angle), np.sin(angle)])*length/2
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


def plot_ray(ray):
    plt.figure()
    plt.plot(ray[:,0],ray[:,1])
    k=5
    plt.plot(k * np.array([1, 1, -1, -1, 1]), k * np.array([1, -1, -1, 1, 1]))
    plt.grid()
    return