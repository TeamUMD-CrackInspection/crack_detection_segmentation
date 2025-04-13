import cv2
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

# 데이터셋 경로
dataset_path = "./datasets/crack-seg"
image_folder = os.path.join(dataset_path, "test/images")  # 학습 이미지 경로
label_folder = os.path.join(dataset_path, "test/labels")  # 학습 라벨 경로

# 이미지 및 라벨 파일 목록 불러오기
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])

# 이미지에 BBox 및 Segmentation Overlay
def visualize(image_path, label_path, alpha=0.4):
    image = cv2.imread(image_path)  # 이미지 로드
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV(BGR) → RGB 변환
    h, w, _ = image.shape  # 원본 이미지 크기

    mask = np.zeros_like(image, dtype=np.uint8)  # 마스크 초기화

    with open(label_path, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        data = line.strip().split()
        class_id = int(data[0])  # 클래스 ID
        polygon = np.array(data[1:], dtype=np.float32).reshape(-1, 2)  # Polygon 좌표
        polygon[:, 0] *= w  # X 좌표 복원
        polygon[:, 1] *= h  # Y 좌표 복원
        polygon = polygon.astype(np.int32)

        # Bounding Box 계산 (Polygon의 최소/최대 좌표 사용)
        x_min, y_min = np.min(polygon, axis=0)
        x_max, y_max = np.max(polygon, axis=0)

        # Bounding Box 그리기
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Segmentation Mask 그리기
        # cv2.polylines(image, [polygon], isClosed=True, color=(255, 0, 0), thickness=1)
        cv2.fillPoly(mask, [polygon], color=(255, 0, 0))  # 투명한 Mask 효과

        text_position = (x_min + 5, y_min + 15)  # 텍스트 위치 (좌상단)
        cv2.putText(image, f"{class_id}:{idx + 1}", text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        
    overlay = cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)

    return overlay



# # 첫 번째 이미지 예제
# image_path = os.path.join(image_folder, image_files[0])
# label_path = os.path.join(label_folder, image_files[0].replace(".jpg", ".txt"))

# output = visualize(image_path, label_path, alpha=0.4)

# plt.figure(figsize=(10, 6))
# plt.imshow(output)
# plt.axis("off")
# plt.show()


# 이미지 인덱스 초기화
index = 0

# Matplotlib figure 설정
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')  # 축 숨기기

def update_image(index):
    """이미지와 마스크를 업데이트하여 Matplotlib에 표시"""
    image_path = os.path.join(image_folder, image_files[index])
    label_path = os.path.join(label_folder, image_files[index].replace(".jpg", ".txt"))
    
    output = visualize(image_path, label_path, alpha=0.3)
    ax.imshow(output)  # 새로운 이미지를 표시
    plt.draw()  # 그림을 갱신

def on_key(event):
    """키보드 이벤트 처리: 왼쪽 화살표(←) 또는 오른쪽 화살표(→)"""
    global index
    if event.key == 'left':  # 이전 이미지로 이동
        index = max(0, index - 1)
        update_image(index)
    elif event.key == 'right':  # 다음 이미지로 이동
        index = min(len(image_files) - 1, index + 1)
        update_image(index)
    elif event.key == 'q':  # 'q' 키로 종료
        plt.close()

# 이미지 처음 로딩
update_image(index)

# 키 이벤트 처리
fig.canvas.mpl_connect('key_press_event', on_key)

# Matplotlib 창을 띄우고 기다리기
plt.show()