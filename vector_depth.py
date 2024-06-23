import numpy as np
import argparse
import os
import cv2
from scipy.ndimage import zoom

def read_marker_info(txt_file):
    with open(txt_file, 'r') as file:
        markers = [list(map(float, line.strip().split())) for line in file.readlines()]
    return markers

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

def draw_depth_vector(image, start_point, z_depth, scale=100):
    end_point = (start_point[0], start_point[1] - int(z_depth * scale))
    cv2.arrowedLine(image, start_point, end_point, (255, 0, 0), 2)  # Draw in red
    return image

parser = argparse.ArgumentParser()
parser.add_argument("--marker_folder", type=str, required=True, help="Path to the folder containing marker information")
parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing image files")
parser.add_argument("--depth_folder", type=str, required=True, help="Path to the folder containing depth files")
parser.add_argument("--depth_format", type=str, choices=['npy', 'png'], required=True, help="Format of the depth files (npy or png)")
parser.add_argument("--visualize", action='store_true', help="Visualize depth vectors on the images")
args = parser.parse_args()

image_width, image_height = 1280, 720
focal_length = 644  # 예시 초점 거리 (픽셀 단위)
cx, cy = 640, 360  # 이미지 중심 (픽셀 단위)

frame_number = 0
while True:
    txt_file_path = os.path.join(args.marker_folder, f"{frame_number:06d}.txt")
    image_file_path = os.path.join(args.image_folder, f"{frame_number:06d}.png")
    depth_file_path = os.path.join(args.depth_folder, f"{frame_number:06d}_depth.{args.depth_format}")
    
    if not (os.path.exists(txt_file_path) and os.path.exists(image_file_path) and os.path.exists(depth_file_path)):
        print(f"No more files to process. Processed {frame_number} frames.")
        break

    image = cv2.imread(image_file_path)

    if args.depth_format == 'npy':
        depth_data = np.load(depth_file_path)
        actual_depth_map = depth_data[0, 0]  # Access the actual depth map
        # Normalize and resize depth data
        depth_max = np.max(actual_depth_map)
        normalized_depth = actual_depth_map / depth_max
        resized_depth_map = zoom(normalized_depth, (720 / normalized_depth.shape[0], 1280 / normalized_depth.shape[1]), order=1)
        reversed_depth_map = 1.0 - resized_depth_map  # Reverse the depth values
    elif args.depth_format == 'png':
        depth_data = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
        reversed_depth_map = depth_data.astype(np.float32)/1000  # Assuming depth.png is in meters

    markers = read_marker_info(txt_file_path)

    marker_positions_3d = []
    for x_norm, y_norm, _ in markers:
        x_pixel = int(x_norm * image_width) 
        y_pixel = int(y_norm * image_height)
        depth_raw = reversed_depth_map[y_pixel, x_pixel]  # Getting depth value from the resized and reversed depth map
        
        x_real = (x_pixel - cx) * (depth_raw / focal_length)
        y_real = (y_pixel - cy) * (depth_raw / focal_length)
        z_real = np.sqrt(depth_raw ** 2 - x_real ** 2 - y_real ** 2)

        print(f"Real-world coordinates at ({x_pixel}, {y_pixel}): ({x_real}, {y_real}, {z_real}) meters")
        marker_positions_3d.append([x_real, y_real, z_real])
        if args.visualize:
            image = draw_depth_vector(image, (x_pixel, y_pixel), depth_raw, scale=100)

    if len(marker_positions_3d) >= 3:
        p1, p2, p3 = np.array(marker_positions_3d[:3])
        normal_vector = calculate_plane_normal(p1, p2, p3)
        centroid_3d = np.mean([p1, p2, p3], axis=0)
        centroid_2d = (int(centroid_3d[0] * focal_length / centroid_3d[2] + cx),
                       int(centroid_3d[1] * focal_length / centroid_3d[2] + cy))

        image = draw_normal_vector(image, centroid_2d, normal_vector[:2], length=100)  # Adjust length as needed
        cv2.imwrite(os.path.join(args.image_folder, f"normal_{frame_number:06d}.png"), image)
    else:
        print(f"Frame {frame_number}: Insufficient marker data for plane calculation.")

    frame_number += 1
