import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path

#chemin_Armand = r"C:\Users\hueta\Downloads\Dataset\continuous_bee_dataset\images"
chemin_Adrian = r"/Users/jonas/Desktop/Dataset/continuous_bee_dataset/images"
#chemin_Gaspard = r"C:\Users\destr\Downloads\Dataset\continuous_bee_dataset\images"

#DATASET = Path(chemin_Armand)
DATASET = Path(chemin_Adrian)
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

I = cv.imread(str(next(DATASET.glob("*.jpg"))), cv.IMREAD_GRAYSCALE)

plt.imshow(I, cmap="gray")
plt.title("Image originale")
plt.axis("off")
plt.show()

# Histogramme (c'est une courbe, pas une image => plt.plot)
hist_I = cv.calcHist([I], [0], None, [256], [0, 256])

plt.plot(hist_I)
plt.xlim([0, 256])
plt.title("Histogramme")
plt.xlabel("Niveau de gris")
plt.ylabel("Nombre de pixels")
plt.show()

# CLAHE
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
I_CLAHE = clahe.apply(I)

plt.subplot(1, 2, 1); plt.imshow(I, cmap="gray"); plt.title("Originale"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(I_CLAHE, cmap="gray"); plt.title("CLAHE"); plt.axis("off")
plt.show()


