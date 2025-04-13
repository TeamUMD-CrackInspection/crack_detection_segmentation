import os

# YOLO 데이터셋 폴더 경로 (labels 폴더와 images 폴더가 포함된 루트 폴더)
# dataset_path = "dataset"
dataset_path = "/home/umd/workspace/crack_detection/data/custom/pattern_cracks_250313/cvat_annotation/split/val"

images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

# labels 폴더가 없다면 생성
os.makedirs(labels_path, exist_ok=True)

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(images_path) if f.endswith(".jpg") or f.endswith(".png")]

for img_file in image_files:
    txt_file = os.path.splitext(img_file)[0] + ".txt"  # 동일한 이름의 .txt 파일
    txt_file_path = os.path.join(labels_path, txt_file)

    # 해당 이미지의 라벨 파일이 없으면 빈 파일 생성
    if not os.path.exists(txt_file_path):
        open(txt_file_path, 'w').close()  # 빈 파일 생성
        print(f"Created empty label file: {txt_file_path}")

print("All missing label files have been created.")

