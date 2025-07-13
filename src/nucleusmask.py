import cv2
import numpy as np

def color_balance(img):
    img = img.astype(np.float32)
    avgR = np.mean(img[:, :, 2])
    avgG = np.mean(img[:, :, 1])
    avgB = np.mean(img[:, :, 0])
    avgGray = (avgR + avgG + avgB) / 3

    img[:, :, 2] *= (avgGray / avgR)
    img[:, :, 1] *= (avgGray / avgG)
    img[:, :, 0] *= (avgGray / avgB)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
  
def rgb_to_cmyk(image):
    image = image.astype(np.float32) / 255.0
    K = 1 - np.max(image, axis=2)
    C = (1 - image[:, :, 2] - K) / (1 - K + 1e-6)
    M = (1 - image[:, :, 1] - K) / (1 - K + 1e-6)
    Y = (1 - image[:, :, 0] - K) / (1 - K + 1e-6)

    C = np.clip(C, 0, 1)
    M = np.clip(M, 0, 1)
    Y = np.clip(Y, 0, 1)
    K = np.clip(K, 0, 1)

    return C, M, Y, K
  
def compute_soft_map(M, K, image):
    # Difference between black and magenta channels
    KM = K - M
    KM = np.clip(KM, 0, 1)

    # HLS + S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2] / 255.0

    # MS and soft map
    MS = np.minimum(M, S)
    soft_map = MS - KM
    soft_map = np.clip(soft_map, 0, 1)

    return soft_map
def threshold_nucleus(soft_map):
    # Shifting to grayscale
    soft_map_uint8 = (soft_map * 255).astype(np.uint8)

    # Frequency histogram
    hist = [0] * 256
    rows, cols = soft_map_uint8.shape
    for r in range(rows):
        for c in range(cols):
            hist[soft_map_uint8[r, c]] += 1

    total = rows * cols
    sum_total = sum(i * hist[i] for i in range(256))

    # Adaptive threshold using variance until segregation point is discovered
    sumB = 0
    wB = 0
    max_var = 0
    threshold = 0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break

        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    # Binary threshold
    binary_mask = np.zeros_like(soft_map_uint8, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if soft_map_uint8[r, c] >= threshold:
                binary_mask[r, c] = 255
            else:
                binary_mask[r, c] = 0
    return binary_mask

def segment_nucleus(image):
    # Color balancing
    balanced = color_balance(image)

    # Shifting from RGB to CMYK Color space
    C, M, Y, K = rgb_to_cmyk(balanced)

    # Generating a soft map
    soft_map = compute_soft_map(M, K, image)

    # Adaptive threshold
    nucleus_mask = threshold_nucleus(soft_map)

    return nucleus_mask
