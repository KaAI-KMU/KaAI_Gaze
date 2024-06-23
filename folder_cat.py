import os
import shutil

def copy_gaze_files(src_root, dest_root):
    for dirpath, dirnames, filenames in os.walk(src_root):
        for filename in filenames:
            if filename == 'Estimated_Gaze_point.csv':
                # Determine the relative path to maintain folder structure
                relative_path = os.path.relpath(dirpath, src_root)
                # Create the corresponding directory in the destination
                dest_dir = os.path.join(dest_root, relative_path)
                os.makedirs(dest_dir, exist_ok=True)
                # Copy the file
                src_file = os.path.join(dirpath, filename)
                dest_file = os.path.join(dest_dir, filename)
                shutil.copy2(src_file, dest_file)
                print(f"Copied {src_file} to {dest_file}")

src_directory = '/home/kaai/kaAI_dataset'
dest_directory = '/home/kaai/kaAI_dataset_gaze'
copy_gaze_files(src_directory, dest_directory)
