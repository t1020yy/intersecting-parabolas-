
from typing import Tuple
import cv2
from matplotlib import pyplot as plt
import numpy as np

from scipy.ndimage import label
from skimage.measure import regionprops

def preprocess_image(img: np.ndarray, threshold, do_morph: bool=False, img_to_sub: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Производит предобработку изображений: вычитание фонового изображения (если задано),
    бинаризацию и морфологическую обработку (если задано)
    '''
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    if img_to_sub is not None:
        res = cv2.subtract(gray, img_to_sub)
    else:
        res = gray
            
    retval, img_binary = cv2.threshold(res, threshold, 255, cv2.THRESH_BINARY)

    if do_morph:
        kernel = np.ones((3,3),np.uint8)
        res = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)

    return res, img_binary

def check_connectivity(img, x, y):
    # 8-connectivity checks
    connectivity = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    count = 0
    for dx, dy in connectivity:
        if 0 <= x+dx < img.shape[1] and 0 <= y+dy < img.shape[0]:
            if img[y+dy, x+dx] > 0:
                count += 1
    return count

def main_processing_loop():
    
    image_path  = 'IMG00169-1.png' 

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # 确认图像已正确加载
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    preproc_img1, binary_img1 = preprocess_image(img_gray, 15, do_morph=False, img_to_sub=None)

    intersection_img = np.zeros_like(binary_img1)
    # 提取图像中所有的白色像素点的坐标
    white_pixels = np.column_stack(np.where(binary_img1 > 0))

    for y, x in white_pixels:
        if check_connectivity(binary_img1, x, y) > 2:
            intersection_img[y, x] = 255

    # Label the intersection points
    labeled_img, num_features = label(intersection_img)
    regions = regionprops(labeled_img)

    # We'll assume the largest connected white region is the intersection area
    intersection_area = max(regions, key=lambda r: r.area)

    # Mark pixels in the intersection area
    for coords in intersection_area.coords:
        intersection_img[coords[0], coords[1]] = 128  # Mark with a different value

    # Now separate the parabolas by removing intersection area
    separated_parabolas_img = np.where(intersection_img == 128, 0, binary_img1)

    # Show the results
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(binary_img1, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(intersection_img, cmap='gray')
    plt.title('Intersection Area')

    plt.subplot(1, 3, 3)
    plt.imshow(separated_parabolas_img, cmap='gray')

    plt.show()


if __name__ == "__main__":

    main_processing_loop()

    