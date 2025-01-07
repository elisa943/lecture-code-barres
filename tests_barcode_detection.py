# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:53:28 2025

@author: Admin
"""
from barcode_detection import *
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TEST DE L'EXTRACTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%
img="img/barcode0.jpg"
# Pour le bruit, à regler à la main (2 est un bon point de départ)
sigma_bruit = 3

# Pour le gradient, relativement faible pour trouver les vecteurs de transition correspondant aux barres
sigma_g = 2

# Pour le tenseur, relativement élevé pour trouer des clusters de vecteurs gradient
sigma_t = 100

seuil = 0.7 
# u=barcode_detection(img,sigma_g,sigma_t,seuil=0.5,sigma_bruit=2)
# u=barcode_detection("img/code_barre_prof.jpg",1,15,0.7,2,affichage=False)
u=barcode_detection("img/barcode0.jpg",sigma_g=5,sigma_t=30,seuil=0.7,sigma_bruit=2)
# u=barcode_detection("img/b1.jpg",2,50,0.7,2,affichage=False)
for blob in u:
    print("Barycentre: ",end="")
    print(blob.barycentre)
    print("Axe: ",end="")
    print(blob.axis)
    print("Valeurs propres: ",end="")
    print(blob.valeurs_propres)
    o=blob.axis
    # blob.__repr__()

print(f"Code exécuté en {round(time()-start_time,3)} s")