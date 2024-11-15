import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import signal
from matplotlib.colors import ListedColormap
from scipy.interpolate import RectBivariateSpline

img_code_barre=plt.imread('')

img_code_barre_YCbCr=skimage.color.rgb2ycbcr(img_code_barre)



plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(img_code_barre)
plt.subplot(1, 2, 2)
plt.imshow(img_code_barre_YCbCr)


plt.show()
