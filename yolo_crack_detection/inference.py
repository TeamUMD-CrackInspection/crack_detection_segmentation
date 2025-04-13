from ultralytics import YOLO
import os
import numpy as np
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.seg_line import save_mask
import json

# Line class -> width 값 매핑
LINE_WIDTH_MAPPING = {
    0: 0.1,
    1: 0.2,
    2: 0.3,
    3: 0.4,
    4: 0.5
}


# IoU 임계값 설정
IOU_THRESHOLD = 0.5

def calculate_iou(box1, box2):
    """ 두 개의 bbox 간 IoU(Intersection over Union)를 계산 """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 교집합 영역 계산
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 각 bbox의 넓이 계산
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 합집합 영역 계산
    union = box1_area + box2_area - intersection

    # IoU 계산
    return intersection / union if union > 0 else 0\
    
def remove_duplication(temp_bboxes):
    
    temp_bboxes.sort(key=lambda x: x[1], reverse=True)  # 신뢰도(conf) 기준 내림차순 정렬
    filtered_bboxes = []

    for bbox in temp_bboxes:
        keep = True
        for filtered in filtered_bboxes:
            iou = calculate_iou(bbox[0], filtered[0])
            if iou > IOU_THRESHOLD:
                keep = False
                break
        if keep:
            filtered_bboxes.append(bbox)

    bboxes = []
    line_classes = []
    # 최종 bbox 리스트 저장
    for bbox, conf, label in filtered_bboxes:
        bboxes.append(bbox)
        line_classes.append(label)
    return bboxes, line_classes


def save_crack_info(bboxes, line_classes, output_json_path):
    cracks_data = {"cracks": []}
    for bbox, line_class in zip(bboxes, line_classes):
        width_value = LINE_WIDTH_MAPPING[line_class]
        cracks_data["cracks"].append({
            "bbox": bbox,  # 튜플을 리스트로 변환하여 저장
            "width": {
                "min": width_value,
                "max": width_value,
                "avg": width_value
            }
        })
    
    # JSON 파일 저장
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(cracks_data, json_file, indent=2)

    print(f"Crack data saved at: {output_json_path}")

def main():

    # 모델 로드 (학습된 가중치 사용)
    model = YOLO("runs/detect/pattern_crack_0313_train/ep150/weights/best.pt")  # 학습된 모델 경로

    # 이미지에서 Inference 실행
    # img_path="/home/umd/workspace/crack_detection/data/public/CrackTree260/image"
    input_folder = "/home/umd/workspace/crack_detection/data/custom/pattern_cracks_250313/cvat_annotation/split/val/images"
    output_folder = "inference_results/pattern_crack_0313_test/ep_100" 

    os.makedirs(output_folder, exist_ok=True)


    for img_name in os.listdir(input_folder):

        if img_name.lower().endswith("png"):
            img_path = os.path.join(input_folder, img_name)
            output_path = os.path.join(output_folder, f"pred_{img_name}")

            results = model(img_path, save=False, imgsz=512, conf=0.5)
            # print("result: ", results)

            # bboxes = []
            # line_classes = []
            temp_bboxes = []

            # 결과 객체 확인
            for i, result in enumerate(results):
                boxes = result.boxes  # Bounding box 정보                
                names = model.names  # Class 이름 정보
                print(f"Detected {len(result.boxes)} objects")
                # result.show()  # 결과 이미지 시각화
                result.save(output_path)  # 결과 저장
                
                for box in boxes:                    
                    xyxy = list(map(int, box.xyxy[0].tolist()))  # (x1, y1, x2, y2) 좌표                    
                    conf = box.conf[0].item()  # 신뢰도(confidence)
                    cls = int(box.cls[0].item())  # 클래스 인덱스
                    label = names[cls]  # 클래스 이름

                    # bboxes.append(xyxy)
                    # line_classes.append(cls)
                    temp_bboxes.append((xyxy, conf, cls))
                    
                    conf_str = f"{conf:.2f}"
                    print(f"Class: {label}:{cls}, BBox: {xyxy}, Confidence: {conf_str}")

                temp_bboxes.sort(key=lambda x: x[1], reverse=True) # 신뢰도(conf) 기준 내림차순 정렬
                bboxes, line_classes = remove_duplication(temp_bboxes)  

                save_mask(img_path, bboxes, os.path.join(output_folder, f"{img_name.split('.')[0]}_mask.png"))
                save_crack_info(bboxes, line_classes, os.path.join(output_folder,f"{img_name.split('.')[0]}_crackinfo.json"))

if __name__ == "__main__":
    main()