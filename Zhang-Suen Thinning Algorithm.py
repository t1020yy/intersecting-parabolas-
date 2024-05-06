from PIL import Image
import cv2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from collections import deque
from skimage.filters import threshold_otsu

def process_and_convert_image(image_path):
    img = Image.open(image_path)
    writable_img = img.copy()
    img_data = writable_img.load()
    width, height = writable_img.size

    for x in range(width):
        for y in range(height):
            pixel = img_data[x, y]
            if pixel == 0:
                img_data[x, y] = 255  # Black to white
            elif pixel == 255:
                img_data[x, y] = 180  # White to gray

    # Convert to numpy array for further processing
    img_np = np.array(writable_img)
    return img_np

# cv2.imshow('Parabola 1', img_np)
def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned

def apply_threshold_and_thinning(image_np):
    Otsu_Threshold = threshold_otsu(image_np)
    BW_Original = image_np < Otsu_Threshold
    BW_Skeleton = zhangSuen(BW_Original)
    return BW_Original, BW_Skeleton


def is_intersection(x, y, image):
    """检查一个像素点是否是交点，中心为黑色且周围全为白色像素"""
    if image[x, y] != 0:  # 确保中心像素是黑色
        return False
    # 获取周围的8个邻居像素
    neighbors = image[x-1:x+2, y-1:y+2]
    # 检查除了中心点之外所有邻居是否都是白色
    neighbors_flattened = neighbors.flatten()
    # 排除中心点，检查其余像素
    return np.all(neighbors_flattened[np.arange(neighbors_flattened.size) != 4] == 255)

def find_and_mark_intersections(BW_Skeleton):
    # binary_img = (BW_Skeleton * 255).astype(np.uint8)
    if BW_Skeleton.dtype != np.uint8:
        BW_Skeleton = (BW_Skeleton * 255).astype(np.uint8)
    _, binary_img = cv2.threshold(BW_Skeleton, 127, 255, cv2.THRESH_BINARY)
    intersections = []

    for x in range(1, binary_img.shape[0] - 1):
        for y in range(1, binary_img.shape[1] - 1):
            if is_intersection(x, y, binary_img):
                intersections.append((x, y))

    color_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for point in intersections:
        cv2.circle(color_img, (point[1], point[0]), 1, (0, 0, 255), -1)  # Red circle marking
        print(intersections[0])
    
    return color_img, intersections

def bfs_track(image, start_points, track_mask):
    queue = deque(start_points)
    output_image = np.zeros_like(image, dtype=np.uint8)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                if image[nx, ny] == 255 and track_mask[nx, ny]:
                    output_image[nx, ny] = 255
                    image[nx, ny] = 0  # Mark this pixel as visited by setting it to black
                    queue.append((nx, ny))
    
    return output_image
image_path = "img.png"
img_np = process_and_convert_image(image_path)
BW_Original, BW_Skeleton = apply_threshold_and_thinning(img_np)
color_img, intersections = find_and_mark_intersections(BW_Skeleton)
cv2.imwrite('marked_image.bmp', color_img)

img_path1 = 'marked_image.bmp'
img = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)

# Assume the intersection point and its neighborhood are determined
intersection_x, intersection_y = intersections[0]  # Example coordinates, adjust as needed
start_points_parabola1 = [(intersection_x - 1, intersection_y - 1), (intersection_x + 1, intersection_y + 1)]
start_points_parabola2 = [(intersection_x - 1, intersection_y + 1), (intersection_x + 1, intersection_y - 1)]

# Mask to track which pixels can still be visited
track_mask = img.copy()
track_mask[track_mask == 255] = True  # Only white pixels are initially visitable

# Track both parabolas
parabola1_img = bfs_track(img.copy(), start_points_parabola1, track_mask)
parabola2_img = bfs_track(img.copy(), start_points_parabola2, track_mask)

# Show and save the results
cv2.imshow('Parabola 1', parabola1_img)
cv2.imshow('Parabola 2', parabola2_img)
cv2.imwrite('parabola_21.bmp', parabola1_img)
cv2.imwrite('parabola_22.bmp', parabola2_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Load the image
# img_path = 'marked_image-2.bmp'
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# # Assume the intersection point and its neighborhood are determined
# intersection_x, intersection_y = 417, 795  # Example coordinates, adjust as needed
# start_points_parabola1 = [(intersection_x - 1, intersection_y - 1), (intersection_x + 1, intersection_y + 1)]
# start_points_parabola2 = [(intersection_x - 1, intersection_y + 1), (intersection_x + 1, intersection_y - 1)]

# # Mask to track which pixels can still be visited
# track_mask = img.copy()
# track_mask[track_mask == 255] = True  # Only white pixels are initially visitable

# # Track both parabolas
# parabola1_img = bfs_track(img.copy(), start_points_parabola1, track_mask)
# parabola2_img = bfs_track(img.copy(), start_points_parabola2, track_mask)

# # Show and save the results
# cv2.imshow('Parabola 1', parabola1_img)
# cv2.imshow('Parabola 2', parabola2_img)
# cv2.imwrite('parabola_21.bmp', parabola1_img)
# cv2.imwrite('parabola_22.bmp', parabola2_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
