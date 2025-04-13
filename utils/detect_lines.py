# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # 이미지 로드
# image_path = "./crack_line.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # 엣지 검출 (Canny 사용)
# edges = cv2.Canny(image, threshold1=50, threshold2=150)

# # 컨투어 찾기
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 원본 이미지 컬러로 변환 (BBox 표시용)
# image_with_bbox = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# # BBox 그리기
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # 결과 이미지 출력
# plt.figure(figsize=(8,6))
# plt.imshow(image_with_bbox, cmap='RdGy')
# plt.title("Detected Lines with BBoxes")
# plt.axis("off")
# plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image_path = "./crack_line.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 이미지 대비 향상 (선 감지 용이하도록)
image_eq = cv2.equalizeHist(image)

# 엣지 검출을 위한 필터 적용 (GaussianBlur + Canny Edge Detection)
blurred = cv2.GaussianBlur(image_eq, (5, 5), 0)
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# 선 검출 (Hough Line Transform)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=5)

# 컬러 이미지로 변환 (시각화 용이하도록)
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# YOLO 라벨 저장할 리스트
labels = []

# 선의 굵기 기반 클래스 매핑 (임의 값 설정, 실제 적용 시 보정 필요)
thickness_classes = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

print("len: ", len(lines))

# 검출된 선을 이미지에 표시
for line in lines:
    x1, y1, x2, y2 = line[0]

    # 선의 두께 추정 (픽셀 강도 차이를 이용, 간단한 방법)
    thickness = abs(y2 - y1) if abs(y2 - y1) > 1 else 1  # 최소 1로 설정

    # 클래스 매핑 (기본값: 가장 가까운 두께 클래스로 할당)
    thickness_class = thickness_classes.get(thickness, 0)

    # Bounding Box 생성
    x_center = (x1 + x2) / (2 * image.shape[1])
    y_center = (y1 + y2) / (2 * image.shape[0])
    width = abs(x2 - x1) / image.shape[1]
    height = thickness / image.shape[0]

    # YOLO 라벨 저장 (클래스 ID, x_center, y_center, width, height)
    labels.append(f"{thickness_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 이미지에 선 표시 (Bounding Box 시각화)
    cv2.rectangle(image_color, (x1, y1 - thickness // 2), (x2, y2 + thickness // 2), (0, 255, 0), 2)

# 라벨 데이터 저장 (파일 생성)
label_path = "./image_labels.txt"
with open(label_path, "w") as f:
    f.write("\n".join(labels))

# 시각화
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("YOLO Bounding Box Visualization")
plt.show()

# 결과 파일 경로 반환
label_path
