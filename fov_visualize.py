import cv2
import pandas as pd
import os

fov_folder = '/home/kaai/yolov5/inference/apgujeong_sample/fov'
gaze_file = '/home/kaai/yolov5/inference/apgujeong_sample/Gaze_point.csv'
output_folder = "/home/kaai/yolov5/inference/apgujeong_sample/fov/val"  # 결과 이미지를 저장할 폴더 경로

os.makedirs(output_folder, exist_ok=True)  # 결과 이미지를 저장할 폴더 생성

gaze_data = pd.read_csv(gaze_file)  # gaze 데이터 파일 읽기

# 이미지 파일의 순서를 가져오기
image_numbers = [int(file.split('.')[0]) for file in os.listdir(fov_folder) if file.endswith(".png")]

for idx, image_number in enumerate(sorted(image_numbers)):
    image_file = f"{image_number:06d}.png"
    image_path = os.path.join(fov_folder, image_file)

    # 출력 추가
    print(f"Processing image from path: {image_path}")

    image = cv2.imread(image_path)  # 이미지 읽기

    if image is None:
        print(f"Failed to read image from path: {image_path}")
        continue

    # 이미지의 크기 가져오기
    image_height, image_width, _ = image.shape

    # 해당 이미지에 대한 gaze 데이터 가져오기
    gaze_x = gaze_data['.norm_pos.x'][idx ]  # idx에 1을 더해 두 번째 행부터 대응
    gaze_y = gaze_data['.norm_pos.y'][idx ]

    # 정규화된 gaze 좌표를 이미지 해상도로 변환
    gaze_x_pixel = int(gaze_x * image_width)
    gaze_y_pixel = int(image_height - gaze_y * image_height)

    # 이미지에 gaze 포인트를 그리기
    cv2.circle(image, (gaze_x_pixel, gaze_y_pixel), 10, (0, 0, 255), -1)  # 빨간색 원 그리기

    # 결과 이미지 저장
    result_image_path = os.path.join(output_folder, image_file)
    cv2.imwrite(result_image_path, image)