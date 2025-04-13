import cv2
import numpy as np
import os
import re

def get_roi_area(roi):
    x_min, y_min = np.min(roi, axis=0)
    x_max, y_max = np.max(roi, axis=0)
    return (x_min, y_min, x_max, y_max)

def is_valid_crop(crop_x, crop_y, crop_w, crop_h, roi_box, min_roi_w, min_roi_h):
    roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_box
    intersect_x_min = max(crop_x, roi_x_min)
    intersect_y_min = max(crop_y, roi_y_min)
    intersect_x_max = min(crop_x + crop_w, roi_x_max)
    intersect_y_max = min(crop_y + crop_h, roi_y_max)
    
    intersect_w = max(0, intersect_x_max - intersect_x_min)
    intersect_h = max(0, intersect_y_max - intersect_y_min)

    print(f"intersect_w, h={intersect_w},{intersect_h}, min_roi_w,h={min_roi_w}, {min_roi_h}")
    
    return intersect_w >= min_roi_w and intersect_h >= min_roi_h

def crop_and_save(image, roi, output_dir, prefix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    x_min, y_min, x_max, y_max = get_roi_area(roi)
    roi_w, roi_h = x_max - x_min, y_max - y_min
    
    shift_x, shift_y = max(50, roi_w // 10), max(50, roi_h // 10)
    # shift_x, shift_y = max(100, roi_w // 5), max(100, roi_h // 5)

    # min_roi_w, min_roi_h = min(100, roi_w // 4), max(100, roi_h // 4)
    min_roi_w, min_roi_h = 150, 150
    
    img_h, img_w = image.shape[:2]
    crop_w, crop_h = 512, 512

    print(f"ROI x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}, w={roi_w}, h={roi_h}, img_w={img_w}, img_h={img_h}")
    
    start_x = x_min + min_roi_w - crop_w
    end_x = x_max - min_roi_w
    start_y = y_min + min_roi_h - crop_h
    end_y = y_max - min_roi_h
    
    count = 0
    skip = 0
    print(f"{start_y}, {end_y}, {shift_y}, {(end_y-start_y)}, {int((end_y-start_y)/shift_y)} / {start_x}, {end_x}, {shift_x}, {(end_x-start_x)}, {int((end_x-start_x)/shift_x)} ")
    for y in range(start_y, end_y, shift_y):
        # print("y:", y)
        for x in range(start_x, end_x, shift_x):            
            # print("x:", x)
            if x < 0 or y < 0 or x + crop_w > img_w or y + crop_h > img_h:
                # print("skip:", skip, x, y, x + crop_w, y + crop_h)
                skip += 1
                continue
            
            if is_valid_crop(x, y, crop_w, crop_h, (x_min, y_min, x_max, y_max), min_roi_w, min_roi_h):
                crop = image[y:y + crop_h, x:x + crop_w]
                crop_filename = os.path.join(output_dir, f'{prefix}_{count:04d}.png')
                cv2.imwrite(crop_filename, crop, [cv2.IMWRITE_PNG_COMPRESSION, 0]) #무손실 압축               
                count += 1
                
    
    print(f"Saved {count} cropped images in {output_dir}")

def select_roi(image):
    roi_points = []
    
    # 화면에 맞게 이미지 크기 축소 (가로 또는 세로 중 긴 쪽을 1000px로 맞춤)
    max_display_size = 1000
    h, w = image.shape[:2]
    resize_factor = max_display_size / max(h, w) if max(h, w) > max_display_size else 1
    resized_image = cv2.resize(image, (int(w * resize_factor), int(h * resize_factor)))

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
            # 선택한 좌표를 원본 크기로 변환
            original_x = int(x / resize_factor)
            original_y = int(y / resize_factor)
            roi_points.append((original_x, original_y))
            cv2.circle(resized_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select ROI", resized_image)
    
    cv2.imshow("Select ROI", resized_image)
    cv2.setMouseCallback("Select ROI", mouse_callback)

    while len(roi_points) < 4:
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    return np.array(roi_points)

def process_images(input_folder, output_folder, prefix_):
    if os.path.isdir(input_folder):
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    else: #file
        image_files = [os.path.basename(input_folder)]        
        input_folder = os.path.dirname(input_folder)  # 파일이 포함된 폴더 경로 추출


    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        print(f"Processing {image_file}")
        
        match = re.search(r'(\d+_\d+)', image_file) # highres_0090_1.bmp -> 0090_1
        if match:
            prefix = prefix_ if prefix_ is not None else match.group(1)


        roi_points = select_roi(image.copy())
        crop_and_save(image, roi_points, os.path.join(output_folder, os.path.splitext(image_file)[0]), prefix)

# Example Usage
# input_folder = "/home/umd/workspace/crack_detection/data/custom/crack_samples/org_with_crack/highres_0008_1_25km_6.4_crop.bmp"  # ROI를 설정할 이미지가 들어 있는 폴더
input_folder = "/home/umd/workspace/crack_detection/data/custom/pattern_cracks/org/highres_0150_1.bmp"  # ROI를 설정할 이미지가 들어 있는 폴더
output_folder = "/home/umd/workspace/crack_detection/data/custom/pattern_cracks/crack_augment/"  # 크롭된 이미지를 저장할 폴더
prefix = None #"008.6.4"
process_images(input_folder, output_folder, prefix)
