"""
Created on Mon Dec  9 13:02:31 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import floor, sqrt
class Blob:
    def __init__(self, pixels=None,X=None,Y=None, barycentre=None, valeurs_propres=None, vecteurs_propres=None,area=None,axis=None):
        self.pixels = pixels 
        self.X = X
        self.Y = Y 
        self.barycentre = barycentre 
        self.valeurs_propres = valeurs_propres 
        self.vecteurs_propres = vecteurs_propres 
        self.area=area
        self.axis=axis
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
        return self.area
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
        if self.axis==None:
            p1 = floor(self.barycentre[1]+self.valeurs_propres[0]/2*self.vecteurs_propres[0][0]), floor(self.barycentre[0]+self.valeurs_propres[0]/2*self.vecteurs_propres[0][1])
            p2 = floor(self.barycentre[1]-self.valeurs_propres[0]/2*self.vecteurs_propres[0][0]), floor(self.barycentre[0]-self.valeurs_propres[0]/2*self.vecteurs_propres[0][1])
            self.axis=[p1,p2]
        return self.axis
        
        
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