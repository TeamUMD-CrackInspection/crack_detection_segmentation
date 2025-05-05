import os
import json
import math

class_to_width = {0: 0.1, 1: 0.2, 2:0.3, 3: 0.4, 4: 0.5}

def load_gt_sorted(gt_path):
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    gt_list = []
    for line in lines:
        values = list(map(float, line.strip().split()))
        if len(values) == 5:
            gt_list.append(values)  # [class, x_center, y_center, width, height]

    gt_processed = []
    
    for gt in gt_list:
        class_id, cx, cy, w, h = gt
        gt_bbox = yolo_to_bbox(cx, cy, w, h, img_width, img_height)
        width = class_to_width[class_id]
        gt_processed.append({
            "bbox": gt_bbox,
            "width": width
        })
    gt_sorted = sorted(gt_processed, key=lambda x: (x["width"], x["bbox"][0]))
    return gt_sorted
       


def load_pred_sorted(pred_path):
    pred_processed = []
    with open(pred_path, 'r') as f:
        predictions = json.load(f)
        for pred in predictions["cracks"]:
            bbox = pred["bbox"]
            width = pred["width"]["avg"]
            pred_processed.append({
                "bbox":bbox,
                "width": width
            })
    pred_sorted = sorted(pred_processed, key=lambda x: (x["width"], x["bbox"][0]))
    return pred_sorted
        
        
        

def load_test_datasets(gt_dir, pred_dir):
    test_datasets = []

    gt_files = sorted(os.listdir(gt_dir))

    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    pred_files = sorted(pred_files)

    # 두 폴더의 파일이 1:1 대응한다고 가정
    for gt_file, pred_file in zip(gt_files, pred_files):
        # 파일 이름을 기준으로 연결된 GT와 Pred 파일을 매칭
        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = os.path.join(pred_dir, pred_file)       

        assert gt_file.replace(".txt", "") == pred_file.replace("_crackinfo.json", "")

        # GT 파일과 Pred 파일을 읽어오기
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            continue

        gt = load_gt_sorted(gt_path)
        pred = load_pred_sorted(pred_path)

        test_datasets.append({
            "gt": gt,
            "pred": pred,
            "name": gt_file.replace(".txt", "")  # 파일 이름에서 공통 부분 추출
        })

    return test_datasets

def yolo_to_bbox(center_x, center_y, width, height, img_width, img_height):
    x1 = int((center_x - width / 2) * img_width)
    y1 = int((center_y - height / 2) * img_height)
    x2 = int((center_x + width / 2) * img_width)
    y2 = int((center_y + height / 2) * img_height)
    return [x1, y1, x2, y2]

def compute_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


def evaluate(gt_list, pred_list, img_width, img_height, iou_thresh=0.5):    
    matched = [False] * len(pred_list)
    results = []

    for gt in gt_list:
        
        gt_bbox = gt["bbox"]
        gt_width_mm = gt["width"]

        best_iou = 0
        best_idx = -1
        for idx, pred in enumerate(pred_list):
            if matched[idx]:
                continue
            iou = compute_iou(gt["bbox"], pred["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_thresh:
            pred_width = pred_list[best_idx]["width"]
            class_match = math.isclose(pred_width, gt_width_mm, abs_tol=0.05)
            matched[best_idx] = True
            results.append({
                "iou": best_iou,
                "class_match": class_match,
                "gt_width": gt_width_mm,
                "pred_width": pred_width,
                "matched": True
            })
        else:
            results.append({
                "iou": best_iou,
                "class_match": False,
                "matched": False
            })

    return results

def compute_metrics(results, num_predictions):
    TP = sum(1 for r in results if r['matched'] and r['class_match'])
    FP = num_predictions - TP
    FN = sum(1 for r in results if not r['matched'])

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

def evaluate_all(test_datasets, img_width, img_height, iou_thresh=0.5):
    all_metrics = []
    
    for i, data in enumerate(test_datasets):
        print(f"\nTest Dataset {i+1} : {data['name']}")
        gt = data["gt"]
        pred = data["pred"]
        # print("gt:", gt)
        # print("pred:", pred)
        if len(gt) == 0 and len(pred) == 0:
            print(f" - (GT/PRED 없음 → 평가 제외)")
            continue        
        

        results = evaluate(gt, pred, img_width, img_height, iou_thresh)
        metrics = compute_metrics(results, len(pred))
        all_metrics.append(metrics)

        for j, res in enumerate(results):
            print(f"  GT {j+1} → Matched: {res['matched']}, IOU: {res['iou']:.2f}, Class Match: {res['class_match']}")
        print("  - Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(metrics["Precision"], metrics["Recall"], metrics["F1-score"]))
    
    # 평균 성능 계산
    total = {
        "TP": sum(m["TP"] for m in all_metrics),
        "FP": sum(m["FP"] for m in all_metrics),
        "FN": sum(m["FN"] for m in all_metrics),
    }
    total["Precision"] = total["TP"] / (total["TP"] + total["FP"]) if (total["TP"] + total["FP"]) else 0.0
    total["Recall"] = total["TP"] / (total["TP"] + total["FN"]) if (total["TP"] + total["FN"]) else 0.0
    total["F1-score"] = 2 * total["Precision"] * total["Recall"] / (total["Precision"] + total["Recall"]) if (total["Precision"] + total["Recall"]) else 0.0

    print("\n평균 성능 지표:")
    print("  - TP: {}, FP: {}, FN: {}".format(total["TP"], total["FP"], total["FN"]))
    print("  - Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}".format(total["Precision"], total["Recall"], total["F1-score"]))

    return all_metrics, total

# gt_dir = "/home/umd/workspace/crack_ws/data/custom/pattern_cracks_250313/cvat_annotation/split/val/labels"
# pred_dir = "/home/umd/workspace/crack_ws/src/crack_detection_segmentation/yolo_crack_detection/inference_results/pattern_crack_0313_test/ep_100"
gt_dir = "/home/umd/workspace/crack_ws/data/custom/pattern_cracks_250429/cvat_annotation/split/test/labels"
pred_dir = "/home/umd/workspace/crack_ws/src/crack_detection_segmentation/yolo_crack_detection/inference_results/pattern_cracks_250429_test/base_yolo11_ep100"

img_width, img_height = 512, 512

# 테스트 데이터셋 로드
test_datasets = load_test_datasets(gt_dir, pred_dir)

# 평가 실행
evaluate_all(test_datasets, img_width, img_height)