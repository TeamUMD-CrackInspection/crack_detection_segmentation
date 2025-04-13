import os
import shutil
import random
import yaml

# 원본 데이터셋 경로 (CVAT에서 받은 폴더)
# dataset_path = "/home/umd/workspace/crack_detection/data/custom/crack_samples/crack_sample_annotation"  
dataset_path = "/home/umd/workspace/crack_detection/data/custom/pattern_cracks_250313/cvat_annotation"
split_path = os.path.join(dataset_path, "split")  # 새로운 split 폴더

# Split 비율 설정
split_ratio = 0.8  # Train 80%, Val 20%

# 기존 데이터 폴더 경로
images_dir = os.path.join(dataset_path, "images", "train")
labels_dir = os.path.join(dataset_path, "labels", "train")

# 새로운 split/train 및 split/val 폴더 생성
for split in ["train", "val"]:
    os.makedirs(os.path.join(split_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(split_path, split, "labels"), exist_ok=True)

# 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

# 데이터 섞고 split
random.shuffle(image_files)
split_idx = int(len(image_files) * split_ratio)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]
print("files:" , len(train_files), len(val_files))
# 파일 복사 
def copy_files(files, split):
    for file in files:
        img_src = os.path.join(images_dir, file)
        lbl_src = os.path.join(labels_dir, file.replace(file.split('.')[-1], "txt"))  # 확장자 변경

        img_dst = os.path.join(split_path, split, "images", file)
        lbl_dst = os.path.join(split_path, split, "labels", os.path.basename(lbl_src))

        shutil.copy(img_src, img_dst)  # 이미지 복사
        if os.path.exists(lbl_src):  # 라벨이 있는 경우만 복사
            shutil.copy(lbl_src, lbl_dst)


copy_files(train_files, "train")
copy_files(val_files, "val")

# 기존 data.yaml 불러와서 수정
original_yaml_path = os.path.join(dataset_path, "data.yaml")
with open(original_yaml_path, "r") as f:
    data_yaml = yaml.safe_load(f)

# train/val 경로 수정
data_yaml["path"] = split_path
data_yaml["train"] = "train/images"
data_yaml["val"] = "val/images"

# 새로운 data.yaml 저장
data_yaml_path = os.path.join(split_path, "data.yaml")
with open(data_yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"Train/Val split 완료! 새로운 데이터셋은 {split_path}에 저장되었습니다.")

