from phase1 import *
from barcode_detection import *
#from barcode_detection import barcode_detection

def main():
    # Détection des barres
    img_name='img/barcode0.jpg'
    # Lecture du code barre et vérification de la validité 
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    
    # Seuillage avec la méthode d'Otsu
    seuil = otsu(img)
    Blobs = barcode_detection(img_name,sigma_g=2,sigma_t=50,seuil=0.7,sigma_bruit=2)
    
    height = img.shape[0]
    width  = img.shape[1]
    # x = [568, 1056]
    # y = [465, 798]
    for blob in Blobs:
        # altérer la méthode de clcul de l'axe si besoin (blob.calc_axis_...)
        x,y = blob.axis
        print("Axe: ",end="")
        print(blob.axis)
        print(phase1(x, y, img, seuil))

    return 

main()