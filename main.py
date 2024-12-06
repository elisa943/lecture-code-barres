from phase1 import *

def main():
    # Détection des barres
    
    # Lecture du code barre et vérification de la validité 
    img = cv2.imread('code_barre.png', cv2.IMREAD_GRAYSCALE)
    
    # Seuillage avec la méthode d'Otsu
    seuil = otsu(img)
    
    height = img.shape[0]
    width  = img.shape[1]
    x = [10, width-10]
    y = [height//2, height//2]
    
    print(phase1(x, y, img, seuil))
    return 

main()