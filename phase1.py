import numpy as np
import matplotlib.pyplot as plt
import cv2

def otsu(img, bins=255, displayHisto=False):
    luminance = None
    
    # Si l'image est en couleur (3 dimensions)
    if len(img.shape) == 3 and img.shape[2] > 1:
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
    
    return level

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
    valeurs_img=np.zeros((len(X), 1))

    for i in range(0,len(X)):
        valeurs_img[i]=(img[Y[i], X[i]]) <= seuil
    i1=0
    i2=0

    for i in range(0,len(valeurs_img)):
        if valeurs_img[i]==1:
            i1=i
            break
    for i in range(len(valeurs_img)-1,1,-1):
        if valeurs_img[i]==1:
            i2=i
            break
    return X[i1],Y[i1],X[i2],Y[i2] # xd,yd,xa,ya

def find_u(xd,yd,xa,ya,img,seuil):
    """On va prendre le multiple u et le signal binaire à analyser"""
    nb_p=np.ceil(distance(xd,yd,xa,ya)).astype(int) # Nombre de points L'
    Nb_points=0 # u * 95
    u=0
    while (Nb_points<=nb_p): # On cherche le plus petit u tq u * 95 > nb_p
        Nb_points+=95
        u+=1
    
    X,Y=echantillonnage(xd,yd,xa,ya,Nb_points)
    valeurs_img=np.zeros((len(X), 1))
    
    for i in range(0,len(X)):
        valeurs_img[i]=(img[Y[i], X[i]]) <= seuil
    
    plt.figure()
    plt.imshow(img, cmap='gray')
    for i in range(len(valeurs_img)):
        if valeurs_img[i]: # Noir
            plt.plot(X[i],Y[i],'r.')
        else: # Blanc
            plt.plot(X[i],Y[i],'b.')
    plt.show()

    return valeurs_img,u #Echantillonnage et binarisation

def separate(segment_seuille,u):
    L=np.zeros((12,u*7))
    start=3*u # Détermine les "gardes" (offset)
    for i in range(0,12):
        if (i==6):
            start=start+5*u
        segment_seuille_temp = segment_seuille[start+i*7*u : start+(i+1)*7*u]

        for j in range(len(L[i])):
            L[i,j] = segment_seuille_temp[j,0]  

    return L

def norme_binaire(liste_binaire,chaine_binaire,u):
    sum=0
    for k in range(0,7):
        for i in range(0,u):
            sum+=(liste_binaire[k*u+i]!=int(chaine_binaire[k]))
    return sum

def compare(region_chiffres_bin,L_the,u):
    normes_codes=len(region_chiffres_bin[0])
    decode=np.zeros((12))
    r=""
    for i in range(0,6):
        r_b=r
        for j in range(0,len(L_the[0])):
            if (normes_codes>=norme_binaire(region_chiffres_bin[i],L_the[0][j],u)):
                normes_codes=norme_binaire(region_chiffres_bin[i],L_the[0][j],u)
                decode[i]=int(j)
                r=r_b+"A"
            if (normes_codes>=norme_binaire(region_chiffres_bin[i],L_the[1][j],u)):
                normes_codes=norme_binaire(region_chiffres_bin[i],L_the[1][j],u)
                decode[i]=int(j)
                r=r_b+"B"
        normes_codes=len(region_chiffres_bin[0])
    for i in range(6,len(region_chiffres_bin)):
        for j in range(0,len(L_the[0])):
            if (normes_codes>=norme_binaire(region_chiffres_bin[i],L_the[2][j],u)):
                normes_codes=norme_binaire(region_chiffres_bin[i],L_the[2][j],u)
                decode[i]=int(j)
        normes_codes=len(region_chiffres_bin[0])
    return decode,r

def first_one(decode,r,tab):
    """ Retourne le premier chiffre """
    for i in range(0,len(tab)):
        if (r==tab[i]):
            return i
    return "ERROR"

def clef_controle(decode):
    # Complément à 10 du dernier chiffre du code barre
    complement = (10 - decode[-1]) % 10
    
    # Somme des chiffres de rangs pairs
    somme_pair = 0
    for i in range(1, len(decode), 2): 
        somme_pair += decode[i-1]

    # Somme des chiffres de rangs impairs
    somme_impair = np.sum(decode) - somme_pair
    
    clef = (somme_impair + 3 * somme_pair) % 10
    return clef == complement

def main(x, y, img, seuil):
    x1 = x[0]
    y1 = y[0]
    x2 = x[1]
    y2 = y[1]
    codage_chiffres_bin = [["0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011"],
                           ["0100111", "0110011", "0011011", "0100001", "0011101", "0111001", "0000101", "0010001", "0001001", "0010111"],
                           ["1110010", "1100110", "1101100", "1000010", "1011100", "1001110", "1010000", "1000100", "1001000", "1110100"]]
    codage_premier_chiffre = ["AAAAAA","AABABB","AABBAB","AABBBA","ABAABB","ABBAAB","ABBBAA","ABABAB","ABABBA","ABBABA"]
    Nb=np.ceil(distance(x1, y1, x2, y2)).astype(int) # Nombre de points
    
    # Binarisation
    xd,yd,xa,ya = find_lim(x1,y1,x2,y2,img,seuil)
    
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.plot([xd, xa], [yd, ya], 'ro-')
    plt.plot([x1, x2], [y1, y2], 'go-')
    plt.show()
    
    # Echantillonage + Binarisation de l'image après seuillage 
    segment_seuillage, u = find_u(xd,yd,xa,ya,img,seuil)
    regions_chiffres_bin = separate(segment_seuillage,u)
    
    # Décodage des regions binaires
    regions_chiffres, sequence_AB = compare(regions_chiffres_bin,codage_chiffres_bin,u)
    print(regions_chiffres,sequence_AB)
    
    # Ajout du premier chiffre
    premier_chiffre = first_one(regions_chiffres,sequence_AB,codage_premier_chiffre)
    code_barre = np.append(premier_chiffre, regions_chiffres)
    
    # Vérification de la clé de contrôle
    if clef_controle(code_barre):
        print("Code barre valide : ", code_barre)
        return code_barre
    else:
        print("Code barre invalide : ", code_barre)
        return None  

if __name__ == "__main__":
    img = cv2.imread('code_barre.png', cv2.IMREAD_GRAYSCALE)
    
    # Seuillage avec la méthode d'Otsu
    seuil = otsu(img)
    
    height = img.shape[0]
    width  = img.shape[1]
    
    x = [10, width-10]
    y = [height//2, height//2]
    
    main(x, y, img, seuil)