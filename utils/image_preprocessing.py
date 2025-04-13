import cv2
import numpy as np

# 이미지 로드
image_path = "./crack_line.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 대비 조정 (CLAHE)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
contrast_img = clahe.apply(image)

# 이진화 (Adaptive Threshold 적용)
binary_img = cv2.adaptiveThreshold(contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

# 엣지 검출 (Canny Edge Detection)
edges = cv2.Canny(binary_img, 50, 150)

# 결과 저장
cv2.imwrite("./contrast.png", contrast_img)
cv2.imwrite("./binary.png", binary_img)
cv2.imwrite("./edges.png", edges)

print("전처리 완료! 대비 조정, 이진화, 엣지 검출된 이미지를 저장했습니다.")
