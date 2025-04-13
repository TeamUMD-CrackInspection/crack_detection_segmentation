import cv2
import numpy as np


def save_mask(image_path, bboxes, output_path):
    """
    여러 개의 bbox 영역에서 선을 segmentation하여 하나의 마스크 이미지로 저장하는 함수.

    Parameters:
        image_path (str): 원본 이미지 경로
        bboxes (list of tuples): 여러 개의 (x1, y1, x2, y2) 형태의 bounding box 좌표 리스트
        output_path (str): 마스크 저장 경로

    Returns:
        None
    """

    # 이미지 로드 (grayscale 변환)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"이미지를 찾을 수 없습니다: {image_path}")

    # 원본 크기와 동일한 빈 마스크 생성
    full_mask = np.zeros_like(image, dtype=np.uint8)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox

        # BBox 영역 크롭
        cropped = image[y1:y2, x1:x2]

        # 가우시안 블러 적용 (노이즈 감소)
        blurred = cv2.GaussianBlur(cropped, (5,5), 0)

        # Otsu's Thresholding 적용
        _, mask = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # # Morphological 연산으로 작은 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)        

        # 전체 마스크에 해당 bbox 영역을 적용
        full_mask[y1:y2, x1:x2] = mask

    # 마스크 저장
    cv2.imwrite(output_path, full_mask)
    print(f"Masked image saved at: {output_path}")
