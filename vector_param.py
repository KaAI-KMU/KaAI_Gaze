import numpy as np
import argparse
import os
import cv2

def read_marker_info(txt_file):
    with open(txt_file, 'r') as file:
        markers = [list(map(float, line.strip().split())) for line in file.readlines()]
    return markers

def estimate_depth(D_real, D_image, f):
    return f * D_real / D_image

def calculate_plane_normal(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    return normal / norm if norm else None

def draw_normal_vector(image, start_point, normal_vector, length=100):
    end_point = (int(start_point[0] + normal_vector[0] * length), 
                 int(start_point[1] + normal_vector[1] * length))
    cv2.arrowedLine(image, start_point, end_point, (0, 255, 0), 2)
    return image

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True, help="볼 마커 정보가 저장된 폴더의 경로")
parser.add_argument("--image_folder", type=str, required=True, help="이미지 파일이 저장된 폴더의 경로")
args = parser.parse_args()

f = 322
D_real = 0.05
image_width, image_height = 1289, 720

frame_number = 0
while True:
    txt_file_path = os.path.join(args.folder, f"{frame_number:06d}.txt")
    image_file_path = os.path.join(args.image_folder, f"{frame_number:06d}.png")  # 이미지 형식에 따라 변경 가능
    if not os.path.exists(txt_file_path) or not os.path.exists(image_file_path):
        print(f"No more files to process. Processed {frame_number} frames.")
        break

    image = cv2.imread(image_file_path)
    markers = read_marker_info(txt_file_path)

    marker_positions_3d = []
    for x_norm, y_norm, r_norm in markers:
        x_pixel = int(x_norm * image_width)
        y_pixel = int(y_norm * image_height)
        D_image_pixels = 2 * r_norm * image_width
        depth = estimate_depth(D_real, D_image_pixels, f)
        marker_positions_3d.append([x_pixel, y_pixel, depth])

    if len(marker_positions_3d) >= 3:
        p1, p2, p3 = np.array(marker_positions_3d[:3])
        normal_vector = calculate_plane_normal(p1, p2, p3)[:2]  # 2D 이미지에 그리기 위해 Z 컴포넌트 제거
        centroid_2d = np.mean([p1[:2], p2[:2], p3[:2]], axis=0).astype(int)
        image = draw_normal_vector(image, tuple(centroid_2d), normal_vector, length=100000)
        cv2.imwrite(os.path.join(args.image_folder, f"normal_{frame_number:06d}.png"), image)  # 결과 이미지 저장
    else:
        print(f"Frame {frame_number}: Insufficient marker data for plane calculation.")

    frame_number += 1
