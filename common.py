# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:58:05 2024

@author: Admin
"""
import numpy as np

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