import numpy as np

def echantillonnage(x1,y1,x2,y2,Nb_points):
    """On échantillone sur le segment (x1,y1)->(x2,y2)"""
    """On va choisir le plus proche voisin"""
    X=np.around(np.linspace(np.floor(x1),np.ceil(x2),Nb_points)).astype(int)
    Y=np.around(np.linspace(np.floor(y1),np.ceil(y2),Nb_points)).astype(int)
    return X,Y

def find_lim(x1,y1,x2,y2,img,seuil):
    """Récupération des points de départ et d'arrivée pour le segment 2"""
    X,Y=echantillonnage(x1,y1,x2,y2,np.ceil(np.sqrt((x2-x1)**2+(y2-y1)**2)).astype(int))
    L=img[X,Y]>=seuil #Binarisation
    i1=0
    i2=0
    for i in range(0,len(L)):
        if L[i]==0:
            i1=i
            break
    for i in range(len(L)-1,1,-1):
        if L[i]==0:
            i2=i
            break
    return X[i1],Y[i1],X[i2],Y[i2] # xd,yd,xa,ya

def find_u(xd,yd,xa,ya,img,seuil):
    """On va prendre le multiple u et le signal binaire à analyser"""
    nb_p=np.ceil(np.sqrt((xa-xd)**2-(ya-yd)**2))
    Nb_points=0
    u=0
    while (Nb_points<nb_p):
        Nb_points+=95
        u+=1
    return img[echantillonnage(xd,yd,xa,ya,Nb_points)]>=seuil,u #Echantillonnage et binarisation


x1=2
x2=5
y1=9
y2=6
Nb=np.ceil(np.sqrt((x2-x1)**2+(y2-y1)**2)).astype(int)
X,Y=echantillonnage(x1,y1,x2,y2,Nb)
print(X,Y)