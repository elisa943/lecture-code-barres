import numpy as np
import matplotlib.pyplot as plt

def otsu(img, bins=255, displayHisto=False):
    luminance = None
    print("Image shape: ", img.shape)
    # Si l'image est en couleur (3 dimensions)
    if len(img.shape) == 3:
        # Calcul de la luminance 
        luminance = np.array([[(img[i][j][0] + img[i][j][1] + img[i][j][2])//3 for j in range(img.shape[1])] for i in range(img.shape[0])])
        luminance = luminance.ravel()
    else:
        luminance = img.ravel()
    
    # Création de l'histogramme
    histogram, bin_edges = np.histogram(luminance.ravel(), range=(0, 255), bins=bins)
    
    # Moyenner l'histogramme
    histogram = histogram/sum(histogram)    
    
    # Création d'un dico pour associer chaque valeur de luminance à sa fréquence
    histogram_dic = {int(bin_edges[i]): histogram[i] for i in range(len(histogram))}
    
    # Initialisation des moyennes et poids initiaux
    n = len(histogram_dic)
    sumB = 0
    wB = 0
    maximum = 0.0
    sum1 = sum(i * histogram_dic[i] for i in range(n))
    total = sum(histogram_dic.values())
    level = 0
    for k in range(1, n):
        wF = total - wB
        if wB > 0 and wF > 0:
            mF = (sum1 - sumB) / wF
            val = wB * wF * (sumB / wB - mF) * (sumB / wB - mF)
            
            if val >= maximum:
                level = k
                maximum = val
        
        wB += histogram_dic[k]
        sumB += (k-1) * histogram_dic[k] # A vérifier si c'est k ou k-1
    
    # Afficher l'histogramme
    if displayHisto:
        plt.figure()
        plt.hist(luminance, bins=bins, range=(0, 255))
        plt.axvline(level, color='r')
        plt.title("Histogramme de la luminance")
        plt.xlabel("Luminance")
        plt.ylabel("Fréquence")
        plt.show()
    
    return histogram, level

def distance(x1,y1,x2,y2):
    """ Calcule la distance entre deux points """
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def echantillonnage(x1,y1,x2,y2,Nb_points):
    """On échantillone sur le segment (x1,y1)->(x2,y2)"""
    """On va choisir le plus proche voisin"""
    X=np.around(np.linspace(np.floor(x1),np.ceil(x2),Nb_points)).astype(int)
    Y=np.around(np.linspace(np.floor(y1),np.ceil(y2),Nb_points)).astype(int)
    return X,Y

def find_lim(x1,y1,x2,y2,img,seuil):
    """Récupération des points de départ et d'arrivée pour le segment 2"""
    X,Y=echantillonnage(x1,y1,x2,y2,np.ceil(distance(x1, y1, x2, y2)).astype(int))
    L = img[Y, X] > seuil #Binarisation
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
    nb_p=np.ceil(distance(xd,yd,xa,ya))
    Nb_points=0
    u=0
    while (Nb_points<nb_p):
        Nb_points+=95
        u+=1
    return img[echantillonnage(xd,yd,xa,ya,Nb_points)]>=seuil,u #Echantillonnage et binarisation

def separate(l_bin,u):
    L=np.zeros(12,u*7)
    start=3*u
    for i in range(0,12):
        if (i==6):
            start=start+5*u
        L[i,:]=l_bin[start+i*7*u:start+(i+1)*7*u]
    return L

def compare(L_exp,L_the):
    min=len(L_exp[0])
    u=len(L_exp[0])/7
    decode=np.zeros(1,12)
    r="000000"
    L_compar=np.copy(L_the)
    for j in range(0,len(L_the[0])):
        L_compar[1,j]=("{0:b}".format(L_the[1,i]).zfill(7))*u
        L_compar[2,j]=("{0:b}".format(L_the[2,i]).zfill(7))*u
        L_compar[3,i]=("{0:b}".format(L_the[3,i]).zfill(7))*u
    for i in range(0,len(L_exp)/2):
        for j in range(0,len(L_compar[0])):
            if (min>sum(L_exp[i]!=L_compar[1,j])):
                min=sum(L_exp[i]!=L_compar[1,j])
                decode[i]=L_the[1,j]
                r[i]='A'
            if (min>sum(L_exp[i]!=L_compar[2,j])):
                min=sum(L_exp[i]!=L_compar[2,j])
                decode[i]=L_the[2,j]
                r[i]='B'
        min=len(L_exp[0])
    for i in range(len(L_exp)/2,len(L_exp)):
        for j in range(0,len(L_compar[0])):
            if (min>sum(L_exp[i]!=L_compar[3,j])):
                min=sum(L_exp[i]!=L_compar[1,j])
                decode[i]=L_the[1,j]
        min=len(L_exp[0])
    return decode,r

def first_one(decode,r,tab):
    for i in range(0,len(tab)):
        if (r==tab[i]):
            decode=[i]+decode
            return decode
    return "ERROR"

def clef_controle(decode):
    # Complément à 10 du dernier chiffre du code barre
    complement = 10 - decode[-1]
    
    # Somme des chiffres de rangs impairs
    somme_impair = 0
    for i in range(1, 12, 2): 
        somme_impair += decode[i-1]

    # Somme des chiffres de rangs pairs
    somme_pair = 0
    for i in range(2, 13, 2):
        somme_pair += decode[i-1]
    
    
    clef = (somme_impair + 3 * somme_pair) % 10
    return clef == complement

def main(x, y, img):
    x1 = x[0]
    y1 = y[0]
    x2 = x[1]
    y2 = y[1]
    codage_chiffres = [[13,25,19,61,35,49,47,59,55,11],
                       [43,51,27,33,29,57,5,17,9,23],
                       [114,102,108,66,92,78,80,68,72,116]]
    codage_premier_chiffre = ["AAAAAA","AABABB","AABBAB","AABBBA","ABAABB","ABBAAB","ABBBAA","ABABAB","ABABBA","ABBABA"]
    Nb=np.ceil(distance(x1, y1, x2, y2)).astype(int) # Nombre de points
    
    # Seuillage avec la méthode d'Otsu
    seuil = otsu(img)
    
    # Echantillonnage
    X, Y = echantillonnage(x1,y1,x2,y2,Nb)
    
    # Binarisation
    xd,yd,xa,ya = find_lim(x1,y1,x2,y2,img,seuil)
    
    # Echantillonage + Binarisation de l'image après seuillage 
    img_seuillage, u = find_u(xd,yd,xa,ya,img,seuil)

    regions_chiffres_bin = separate(img_seuillage,u)
    
    # Décodage des regions binaires
    regions_chiffres, sequence_AB = compare(regions_chiffres_bin,codage_chiffres)
    
    # Ajout du premier chiffre
    regions_chiffres = first_one(regions_chiffres,sequence_AB,codage_premier_chiffre)
    
    # Vérification de la clé de contrôle
    if clef_controle(regions_chiffres):
        print("Code barre valide : ", regions_chiffres)
        return regions_chiffres
    else:
        print("Code barre invalide")

if __name__ == "__main__":
    img = plt.imread('code_barre.png')
    
    height, width, _ = img.shape
    x = [10, width-10]
    y = [10, height-10]
    
    main(x, y, img)

"""
x1=2
x2=5
y1=9
y2=6
Nb=np.ceil(distance(x1, y1, x2, y2)).astype(int)
X,Y=echantillonnage(x1,y1,x2,y2,Nb)
print(X,Y)
codage_chiffres = [[13,25,19,61,35,49,47,59,55,11],[43,51,27,33,29,57,5,17,9,23],[114,102,108,66,92,78,80,68,72,116]]
codage_premier_chiffre = ["AAAAAA","AABABB","AABBAB","AABBBA","ABAABB","ABBAAB","ABBBAA","ABABAB","ABABBA","ABBABA"]
"""