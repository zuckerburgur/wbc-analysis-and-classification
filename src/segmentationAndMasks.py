import cv2
import numpy as np

def ccn8(img):
    eq_list = {}
    label = 10
    dims = img.shape
    rows, cols = dims
    a = np.zeros((rows, cols), dtype=np.uint16)

    # First pass: Initial labeling
    for r in range(rows):
        for c in range(cols):
            if img[r, c] == 0:  # Ignoring background
                a[r, c] = 0
            if img[r, c] > 0:  # Define foreground pixel condition
                top = a[r - 1, c] if r > 0 else 0
                left = a[r, c - 1] if c > 0 else 0
                top_left = a[r - 1, c - 1] if (r > 0 and c > 0) else 0
                top_right = a[r - 1, c + 1] if (r > 0 and c < cols - 1) else 0

                # Collect nonzero neighbors
                neighbors = [top, left, top_left, top_right]
                neighbors = [n for n in neighbors if n > 0]  # Remove background

                if not neighbors:
                    a[r, c] = label
                    eq_list[label] = label
                    label += 2
                else:
                    min_label = min(neighbors)
                    a[r, c] = min_label

                    # Update equivalence for all neighboring labels
                    for n in neighbors:
                        eq_list[n] = min(eq_list.get(n, n), min_label)
            else:
                a[r, c] = img[r, c]

    # Second pass: Update equivalency list
    for r in range(rows):
        for c in range(cols):
            a[r, c] = eq_list.get(a[r, c], a[r, c])

    # Iterative pass to propagate equivalence correctly
    for r in range(rows):
        for c in range(cols):
            if a[r, c] != 0:
                while a[r, c] != eq_list.get(a[r, c], a[r, c]): 
                    eq_list[a[r, c]] = eq_list.get(eq_list[a[r, c]], eq_list[a[r, c]])
                    a[r, c] = eq_list[a[r, c]]

    unique_labels, counts = np.unique(a, return_counts=True)

    print("Number of distinct objects:", len(np.unique(a)))
    print(np.unique(a))
    
    plt.figure(figsize=(6, 6))
    plt.imshow(a, cmap='jet', interpolation='nearest')
    plt.title("Relabeled Image")
    plt.axis("off")  
    plt.show()

    return a
  
def largest_object(labelled_img, min_ratio, fill_value=255):
    unique_labels, counts = np.unique(labelled_img, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    label_counts.pop(0, None)  # Remove background

    if not label_counts:
        return np.zeros_like(labelled_img, dtype=np.uint8)

    # Find largest object area
    largest_area = max(label_counts.values())

    # Filter: keep largest + all others >= min_ratio of largest
    filtered_mask = np.zeros_like(labelled_img, dtype=np.uint8)
    for label, area in label_counts.items():
        if area >= min_ratio * largest_area:
            filtered_mask[labelled_img == label] = fill_value
    return filtered_mask
  
def dilation(img, mask_size, se_option):
    rows, cols = img.shape
    midval = mask_size // 2
    bgval = img[3, 3]

    if se_option == 1:
        ifilter = cv2.getStructuringElement(cv2.MORPH_RECT, (mask_size, mask_size))
    elif se_option == 2:
        ifilter = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_size, mask_size))
    elif se_option == 3:
        ifilter = cv2.getStructuringElement(cv2.MORPH_CROSS, (mask_size, mask_size))
    elif se_option == 4:
        ifilter = np.zeros((mask_size, mask_size), dtype=np.uint8)
        for r in range(mask_size):
            for c in range(mask_size):
                if abs(r - midval) + abs(c - midval) <= midval:
                    ifilter[r, c] = 1
    else:
        print("Invalid SE option.")

    padded = np.pad(img, pad_width=midval, mode='constant', constant_values=bgval)
    output = np.zeros_like(img)

    for r in range(rows):
        for c in range(cols):
            sub_image = padded[r:r + mask_size, c:c + mask_size]
            if np.any(sub_image[ifilter == 1]==255):
                output[r, c] = 255
    #
    # cv2.imshow("Dilated Image", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return output

def conv(img, mask_size, mask_val):
    img = img.astype(np.uint8)
    rows, cols, channels = img.shape
    midval = mask_size // 2  # Padding size
    ifilter = np.ones((mask_size, mask_size), dtype=np.float32) * (mask_val / (mask_size ** 2))
    output_image = np.zeros((rows, cols, channels), dtype=np.uint8)

    for ch in range(channels):
        padded = np.pad(img[:, :, ch], midval, mode='constant')
        for r in range(rows):
            for c in range(cols):
                sub_image = padded[r:r + mask_size, c:c + mask_size]
                result = np.sum(sub_image * ifilter)
                output_image[r, c, ch] = np.clip(result, 0, 255)

    return output_image


def mask_outline(mask, mask_size, se_option):
    # Dilating the nucleus
    dilated = dilation(mask, mask_size, se_option)

    # Subtracting original nucleus mask to get the outline
    roc = cv2.subtract(dilated, mask)
    return dilated, roc
  
def cytoplasm_circle(nucleus_mask, padding):
    ys, xs = np.where(nucleus_mask == 255)

    if len(xs) == 0 or len(ys) == 0:
        print("Empty nucleus mask")
        return np.zeros_like(nucleus_mask)
      
    # Mask Center
    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))

    # Radius of the circle = max radial distance from center
    distances = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
    max_radius = int(np.max(distances)) + padding

    # Filled circle for mask
    circular_mask = np.zeros_like(nucleus_mask, dtype=np.uint8)
    cv2.circle(circular_mask, (center_x, center_y), max_radius, 255, thickness=-1)

    return circular_mask
  
