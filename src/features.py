import cv2
import numpy as np

def extract_features(img, nMask, nOutline, cyto, cOutline):
    features = {}

    # Nucleus area
    ncoords = np.column_stack(np.where(nMask == 255))
    nArea = len(ncoords)
    features['nucleus area'] = nArea

    # Cytoplasm area
    coords = np.column_stack(np.where(cyto == 255))
    cArea = len(coords)
    features['cyto area'] = cArea

    # Nucleus Perimeter
    pcoords = np.column_stack(np.where(nOutline == 255))
    nPerimeter = len(pcoords)
    features['nucleus perimeter'] = nPerimeter

    # Cyto Perimeter
    cpcoords = np.column_stack(np.where(cOutline == 255))
    cPerimeter = len(cpcoords)
    features['cyto perimeter'] = cPerimeter

    # Circularity
    if nPerimeter != 0:
        features['circularity'] = round((nPerimeter ** 2)/(4 * np.pi * nArea), 4)
    else:
        features['circularity'] = 0

    # Solidity = area of nuceus / area of cell
    solidity = round(float(nArea / cArea) , 4)
    features['solidity'] = solidity if cArea != 0 else 0

    # Means of red and blue channels
    masked_pixels = img[nMask == 255]
    mean_r = np.mean(masked_pixels[:, 0])
    mean_b = np.mean(masked_pixels[:, 2])

    features['mean_r'] = round(mean_r, 4)
    features['mean_b'] = round(mean_b, 4)
  
    return features
