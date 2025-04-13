import cv2
import numpy as np

def polygon_to_bbox(polygon):
    """
    Polygon 좌표를 받아서 Bounding Box 좌표로 변환
    polygon: [[x1, y1], [x2, y2], ..., [xn, yn]] 형태의 리스트
    return: (x_min, y_min, x_max, y_max)
    """
    polygon = np.array(polygon)
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)

    return x_min, y_min, x_max, y_max

# # 예제: Crack Polygon 좌표
# polygon = [[100, 200], [120, 220], [130, 210], [110, 190]]
# bbox = polygon_to_bbox(polygon)
# print("Bounding Box:", bbox)  # (100, 190, 130, 220)


def polygon_to_mask(polygon_coords, w=512, h=512):
    # 2. Binary Mask로 변환 (512x512 이미지 기준)
    mask_shape = (w, h)
    binary_mask = np.zeros(mask_shape, dtype=np.uinmask5)
#    binary_mask = np.zeros(mask_shape, dtype=np.uint8)
#    cv2.fillPoly(binary_mask, polygon_coords, 255)    

    return binary_mask
