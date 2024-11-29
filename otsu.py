import matplotlib.pyplot as plt
import numpy as np 
from skimage.filters import threshold_otsu

img = plt.imread('code_barre.png')

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
    
    print("Seuil optimal: ", level)
    
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

def displayOtsu(img, OtsuFunc=True):
    thresh = None 
    histogram = None
    
    if (OtsuFunc):
        histogram, thresh = otsu(img)
    else: 
        thresh = threshold_otsu(img)
    
    binary = img > thresh
    
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(binary, cmap=plt.cm.gray)
    plt.title("Seuillage")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title("Image originale")
    plt.axis('off')

    plt.show()

displayOtsu(img)