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

chemin = next(DATASET.glob("*.jpg"))   #première image du dataset
img = cv.imread(str(chemin), cv.IMREAD_GRAYSCALE)

plt.imshow(img, cmap="gray")
plt.title(chemin.name)
plt.axis("off")
plt.show()