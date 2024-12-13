from phase1 import *
from barcode_detection import barcode_detection

def main():
    # Détection des barres
    
    # Lecture du code barre et vérification de la validité 
    img = cv2.imread('img/barcode0.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Seuillage avec la méthode d'Otsu
    seuil = otsu(img)
    
    height = img.shape[0]
    width  = img.shape[1]
    x = [568, 1056]
    y = [465, 798]
    
    print(phase1(x, y, img, seuil))
    return 

main()