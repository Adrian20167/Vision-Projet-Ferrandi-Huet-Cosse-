import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path

chemin_Armand = r"C:\Users\hueta\Downloads\Dataset\continuous_bee_dataset\images"
#chemin_Adrian = r""
#chemin_Gaspard = r""

DATASET = Path(chemin_Armand)
#DATASET = Path(chemin_Adrian)
#DATASET = Path(chemin_Gaspard)

#Permet d'afficher une ligane
"""
chemin = next(DATASET.glob("*.jpg"))   #première image du dataset
img = cv.imread(str(chemin), cv.IMREAD_GRAYSCALE)

plt.imshow(img, cmap="gray")
plt.title(chemin.name)
plt.axis("off")
plt.show()
"""

#On veut appliquer un filtre sur les images, pour faire des tests on prends la première image du dataset
I = cv.imread(next(DATASET.glob("*.jpg")))

plt.imshow(I , cmap="gray")
plt.title("Image originale")
plt.axis("off")
plt.show()

#On veut tester 2 choses, ameliorer le constrast et faire disparaître la texture des alvéoles

#Améliorer le contraste

#On veut l'histogramme

hist_I = cv.calcHist([I], [0], None, [256], [0, 256])
#Attebtion: hist_I est une courbe pas une image => plt.plot()

plt.plot(hist_I)
plt.xlim([0, 256])
plt.title("Histogramme")
plt.xlabel("Niveau de gris")
plt.ylabel("Nombre de pixels")
plt.show()

