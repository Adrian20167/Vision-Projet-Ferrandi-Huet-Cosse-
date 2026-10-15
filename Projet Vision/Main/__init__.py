import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path

chemin_Armand = r"C:\Users\hueta\Downloads\Dataset\continuous_bee_dataset\images"
#chemin_Adrian = r"/Users/jonas/Desktop/Dataset/continuous_bee_dataset/images"
#chemin_Gaspard = r"C:\Users\destr\Downloads\Dataset\continuous_bee_dataset\images"

DATASET = Path(chemin_Armand)
#DATASET = Path(chemin_Adrian)
#DATASET = Path(chemin_Gaspard)

I = cv.imread(str(next(DATASET.glob("*.jpg"))), cv.IMREAD_GRAYSCALE)


"""
# Figure 1 : image originale
plt.figure("Image originale")
plt.imshow(I, cmap="gray")
plt.title("Image originale")
plt.axis("off")
"""

# Figure 2 : histogramme
hist_I = cv.calcHist([I], [0], None, [256], [0, 256])

plt.figure("Histogramme")
plt.plot(hist_I)
plt.xlim([0, 256])
plt.title("Histogramme")
plt.xlabel("Niveau de gris")
plt.ylabel("Nombre de pixels")

# Figure 3 : CLAHE et son histogramme
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
I_CLAHE = clahe.apply(I)
hist_I_CLAHE = cv.calcHist([I_CLAHE], [0], None, [256], [0, 256])

plt.figure("CLAHE")
plt.imshow(I_CLAHE, cmap="gray"); plt.title("Image CLAHE"); plt.axis("off")
#plt.plot(hist_I_CLAHE); plt.xlim([0, 256]); plt.title("Histogramme CLAHE"); plt.xlabel("Niveau de gris"); plt.ylabel("Nombre de pixels")


# C'est nul
# Spectre de Fourier
F = np.fft.fft2(I)
F_centre = np.fft.fftshift(F)
spectre = np.log(1 + np.abs(F_centre))

plt.figure("Spectre de Fourier")
plt.imshow(spectre, cmap="gray")
plt.title("Spectre de Fourier")

# Distance entre le centre et un point lumineux du spectre : environ 105 pixels
d_pic = 105
d = d_pic / 3          # écart-type du gaussien, bien inférieur à la distance du pic

# Filtre passe-bas gaussien
h, w = I.shape
y, x = np.ogrid[:h, :w]
distance2 = (x - w // 2) ** 2 + (y - h // 2) ** 2

masque = np.exp(-distance2 / (2 * d ** 2))

F_filtre = F_centre * masque
I_filtree = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtre)))
I_filtree = np.clip(I_filtree, 0, 255).astype(np.uint8)

#Affichage du filtre

plt.figure("Image filtrée")
plt.imshow(I_filtree, cmap="gray"); plt.title(f"Passe-bas gaussien, d = {d:.0f}"); plt.axis("off")


plt.show()

