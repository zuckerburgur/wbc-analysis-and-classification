import cv2
import numpy as np

def classify_wbc(features):
    nArea = features['nucleus area']
    cArea = features['cyto area']
    solidity = features['solidity']
    circularity = features['circularity']
    nucleus_perim = features['nucleus perimeter']
    mean_r = features['mean_r']
    mean_b = features['mean_b']


    # Adjust values based on observed thresholds from plots for better accuracy while classification  
    if nArea > 35000 and solidity > 0.45:
        return 'Basophil'
    elif nucleus_perim < 3000 and mean_r < 130 and circularity < 40:
        return 'Lymphocyte'
    elif mean_r > 125 and  cArea > 54000 and mean_b > 120:
        return 'Monocyte'
    elif mean_r > 105 and nArea > 18000 and mean_b > 105:
        return 'Eosinophil'
    else:
        return 'Neutrophil'
