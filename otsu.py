import matplotlib.pyplot as plt
import numpy as np 
from skimage.filters import threshold_otsu

img = plt.imread('../img/bdx.jpg')

def otsu(img, bins=256, displayHisto=False):
    # Calcul de la luminance 
    luminance = np.array([[(img[i][j][0] + img[i][j][1] + img[i][j][2])//3 for j in range(img.shape[1])] for i in range(img.shape[0])])
    luminance = luminance.ravel()
    
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
    level = -1 
    for ii in range(1, n):
        wF = total - wB
        if wB > 0 and wF > 0:
            mF = (sum1 - sumB) / wF
            val = wB * wF * (sumB / wB - mF) * (sumB / wB - mF)
            
            if val >= maximum:
                level = ii
                maximum = val
        
        wB += histogram_dic[ii]
        sumB += (ii-1) * histogram_dic[ii]
    
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
    
    return level


# https://en.wikipedia.org/wiki/Otsu%27s_method
def otsu2(img, bins=256, displayHisto=False):
    # Calcul de la luminance 
    luminance = np.array([[(img[i][j][0] + img[i][j][1] + img[i][j][2])//3 for j in range(img.shape[1])] for i in range(img.shape[0])])
    luminance = luminance.ravel()
    
    # Création de l'histogramme
    histogram, bin_edges = np.histogram(luminance.ravel(), range=(0, 255), bins=bins)

    # Moyenner l'histogramme
    histogram = histogram/sum(histogram)    
    
    # Création d'un tableau pour associer chaque valeur de luminance à sa fréquence
    histogram_tab = [[int(bin_edges[i]), histogram[i]] for i in range(len(histogram))]

    # Initialisation des moyennes et poids initiaux
    n = len(histogram_tab)
    print(n)
    sigma_b_carre = np.zeros((n-1, 1))
    w_i  = np.zeros((n, 2))
    mu_i = np.zeros((n, 2))
    numerateur = [i*histogram_tab[i][1] for i in range(n)] # le ième élément étant i * p(i)
    
    # Numérateurs des moyennes mu_i_0 et mu_i_1
    mu_i_0 = numerateur[0]              
    mu_i_1 = sum(numerateur) - mu_i_0

    w_i[0][0]   = histogram_tab[0][1]
    w_i[0][1]   = sum(histogram_tab[i][1] for i in range(1, n))    
    mu_i[0][0]  = numerateur[0] 
    mu_i[0][1]  = sum(numerateur) - mu_i[0][0]

    for t in range(1, n):
        w_i[t][0]   = w_i[t-1][0] + histogram_tab[t][1]
        w_i[t][1]   = w_i[t-1][1] - histogram_tab[t][1]
        mu_i[t][0]  = mu_i_0/w_i[t][0]
        mu_i[t][1]  = mu_i_1/w_i[t][1]

        sigma_b_carre[t-1] = w_i[t][0] * w_i[t][1] * np.power(mu_i[t][0] - mu_i[t][1], 2)
        
        mu_i_0 += numerateur[t]
        mu_i_1 -= numerateur[t]
    
    max_index = np.argmax(sigma_b_carre)
    
    print(sigma_b_carre)
    #print(mu_i[1][0], numerateur[0], w_i[1][0])
    print("Seuil optimal: ", max_index + 1)
    
    # Afficher l'histogramme
    if displayHisto:
        plt.figure()
        #plt.hist(luminance, bins=bins, range=(0, 255))
        plt.plot(histogram)
        plt.plot(sigma_b_carre/(np.sum(sigma_b_carre)), color='r')
        plt.title("Histogramme de la luminance")
        plt.xlabel("Luminance")
        plt.ylabel("Fréquence")
        plt.show()
    
    
    return histogram, max_index + 1


def displayOtsu(img, OtsuFunc=True):
    thresh = None 
    histogram = None
    if (OtsuFunc):
        histogram, thresh = otsu(img)
    else: 
        thresh = threshold_otsu(img)
    
    binary = img > thresh
    
    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.imshow(binary, cmap=plt.cm.gray)
    plt.title("Seuillage")

    plt.subplot(1, 3, 2)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title("Image originale")
    
    plt.subplot(1, 3, 3)
    plt.hist(img.ravel(), bins=256, range=(0, 255))
    plt.axvline(thresh, color='r')
    plt.title("Histogramme")

    plt.show()

otsu(img, displayHisto=True)