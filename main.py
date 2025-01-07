from phase1 import *
from barcode_detection import *
start=time()
# Détection des barres
# img_name = 'img/barcode0.jpg'
img_name = 'img/code_barre_prof.jpg'
# Lecture du code barre et vérification de la validité
img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

# Seuillage avec la méthode d'Otsu
seuil = otsu(img)

# Extraction des points d'intérêt
Blobs = barcode_detection(img_name, sigma_g=1, sigma_t=15, seuil=0.7, sigma_bruit=2)
# %%
height = img.shape[0]
width = img.shape[1]

for blob in Blobs:
    # altérer la méthode de calcul de l'axe si besoin (blob.calc_axis_...)
    # blob.calc_axis_ray(len_adjust=5)
    
    x = blob.axis[:,0].astype(int).tolist()[::-1]
    y = blob.axis[:,1].astype(int).tolist()[::-1]
    print("Axe: ", end="")
    print(x,y)
    print(phase1(x, y, img, seuil))

duration=time()-start
print("code exécuté en {}s".format(round(duration,2)))