import os
import subprocess
from pathlib import Path
import argparse
import shutil

def run_detection(source_dir, weights, conf_threshold=0.4):
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            head_color_path = os.path.join(root, dir_name, "head_color")
            if os.path.exists(head_color_path):
                save_dir = os.path.join(root, dir_name, "ballmarker_labels")
                images_dir = os.path.join(save_dir, "images")
                labels_dir = os.path.join(save_dir, "labels")
                
                # 기존 디렉토리 삭제
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)
                
                command = [
                    "python", "detect.py",
                    "--source", head_color_path,
                    "--weights", weights,
                    "--conf-thres", str(conf_threshold),
                    "--save-ball",
                    "--project", save_dir,
                    "--name", "", 
                    "--exist-ok"
                ]
                
                print(f"Running command: {' '.join(command)}")
                subprocess.run(command)

                # Move the generated files to the correct directories
                for file_name in os.listdir(save_dir):
                    if file_name.endswith(".png"):
                        os.rename(os.path.join(save_dir, file_name), os.path.join(images_dir, file_name))
                    elif file_name.endswith(".txt"):
                        os.rename(os.path.join(save_dir, file_name), os.path.join(labels_dir, file_name))

def run_vector_depth_rs(dataset_dir):
    for root, dirs, files in os.walk(dataset_dir):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            marker_folder = os.path.join(folder_path, "ballmarker_labels", "labels")
            image_folder = os.path.join(folder_path, "ballmarker_labels", "images")
            depth_folder = os.path.join(folder_path, "head_depth")
            lidar_folder = os.path.join(folder_path, "lidar")
            gaze_file = os.path.join(folder_path, "Gaze_point.csv")
            front_image = os.path.join(folder_path, "flir4")
            
            if all(os.path.exists(path) for path in [marker_folder, image_folder, depth_folder, lidar_folder, gaze_file, front_image]):
                command = [
                    "python", "vector_depth_rs.py",
                    "--marker_folder", marker_folder,
                    "--image_folder", image_folder,
                    "--depth_folder", depth_folder,
                    "--lidar_folder", lidar_folder,
                    "--depth_format", "png",
                    "--gaze_file", gaze_file,
                    "--front_image", front_image
                ]
                
                print(f"Running command: {' '.join(command)}")
                subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv5 detection and vector_depth_rs on dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--weights", type=str, default="runs/train/exp3/weights/best.pt", required=True, help="Path to the weights file")
    parser.add_argument("--conf-thres", type=float, default=0.4, help="Confidence threshold for detection")

    args = parser.parse_args()

    #run_detection(args.dataset, args.weights, args.conf_thres)
    run_vector_depth_rs(args.dataset)
