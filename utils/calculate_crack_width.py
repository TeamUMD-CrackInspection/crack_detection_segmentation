import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from convert_utils import polygon_to_mask

def load_mask(mask_image_path):
    # 파일 로드
    # image_path = "/mnt/data/image.png"
    crack_mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # 이진화 (흰색 crack을 검은 배경에서 분리)
    _, binary_mask = cv2.threshold(crack_mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask




def calculate_width(binary_mask):
    # Skeletonization (골격화)
    skeleton = skeletonize(binary_mask // 255)

    # 거리 변환 (각 픽셀에서 가장 가까운 0 픽셀까지의 거리 계산)
    distance_map = distance_transform_edt(binary_mask)512

    print("....", distance_map.shape)

    # Skeleton에서 거리 값 추출 (crack의 중심선에서의 거리값 * 2 = 두께)
    thickness_values = distance_map[skeleton > 0] * 2

    # 두께의 평균, 최소, 최대값 계산
    thickness_mean = np.mean(thickness_values)
    thickness_min = np.min(thickness_values)
    thickness_max = np.max(thickness_values)

    plot_results(binary_mask, skeleton, distance_map)

    # 결과 출력
    return thickness_mean, thickness_min, thickness_max

def plot_results(binary_mask, skeleton, distance_map):
    # # 결과 시각화
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(binary_mask, cmap='gray')
    # plt.title("Binary Mask from Polygon")

    # plt.subplot(1, 2, 2)
    # plt.imshow(skeleton, cmap='gray')
    # plt.title("Skeletonized Crack")

    # plt.show()

    # Skeleton과 거리 변환 맵 시각화
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(binary_mask, cmap='gray')
    ax[0].set_title("Binary Mask")

    ax[1].imshow(skeleton, cmap='gray')
    ax[1].set_title("Skeleton")

    ax[2].imshow(distance_map, cmap='jet')
    ax[2].set_title("Distance Transform")

    plt.show()


def main():
    input_type = "mask" #polygon
    if input_type == "mask":
        mask_image_path = "/home/umd/workspace/crack_detection/data/public/CrackVision12K/split_dataset_final/test/GT/10810.png"
        binary_mask = load_mask(mask_image_path)
    else:
        # 1. Polygon 데이터 예제 (YOLO Segmentation 형식)
        polygon_coords = [  # 예제 좌표
            np.array([[50, 100], [150, 80], [200, 120], [180, 200], [60, 180]], np.int32)
        ]
        polygon_to_mask(polygon_coords)

    mean, min, max = calculate_width(binary_mask)

    print(f"width mean:{mean:.2f}px, min:{min:.2f}px, max:{max:.2f}")

    
if __name__ == "__main__":
    main()




