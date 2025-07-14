from src.classification import classify_wbc
from src.features import extract_features
from src.nucleusmask import color_balance, rgb_to_cmyk, compute_soft_map, threshold_nucleus, segment_nucleus
from src.segmentationAndMasks import ccn8, largest_object, dilation, conv, mask_outline, cytoplasm_circle

import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
  image = cv2.imread('image_path.jpg') # Subsitute with your actual image path
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  blurred = conv(image, 7, 1)
  nucleus_mask = segment_nucleus(blurred)
  
  # Invert the nucleus mask (so holes become white)
  inv = cv2.bitwise_not(nucleus_mask)
  
  # Applying CCA to detect holes
  labeled = ccn8(inv)
  num_labels = len(np.unique(labeled))
  nucleus_area = np.sum(nucleus_mask == 255)
  for label in range(1, num_labels):
      component_area = np.sum(labeled == label)
      if component_area / nucleus_area < 0.4:
          ccaNucleus = ccn8(nucleus_mask)
          nucleus_mask = largest_object(ccaNucleus, 0.4)
          break
  
  
  roc = mask_outline(nucleus_mask, 5, 2)
  cell = cytoplasm_circle(roc, 10)
  cytor = mask_outline(cell, 5, 2)
  
  cv2.imshow("Cell", cell)
  cv2.imshow("Nucleus Mask", nucleus_mask)
  cv2.imshow("Nucleus Outline", roc)
  cv2.imshow("Cell Outline", cytor)
  
  
  features = extract_features(image, nucleus_mask, roc, cell, cytor)
  classify_wbc(features)

  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
    main()

