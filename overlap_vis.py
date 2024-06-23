import cv2
import numpy as np

# Load the depth map image
depth_map_path = "/home/kaai/yolov5/runs/detect/exp14/sample/depth/000000.png"
depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

# Clip the depth map values to 0-2 meters and normalize
depth_map_clipped = np.clip(depth_map, 0, 2000)  # Assuming depth is in millimeters, clip to 2 meters
normalized_depth_map = cv2.normalize(depth_map_clipped, None, 0, 255, cv2.NORM_MINMAX)
normalized_depth_map = np.uint8(normalized_depth_map)

# Apply viridis color map
viridis_colormap = cv2.applyColorMap(normalized_depth_map, cv2.COLORMAP_VIRIDIS)

# Save the result
output_path = "/home/kaai/yolov5/runs/detect/exp14/sample/depth/viridis_colormap.png"
cv2.imwrite(output_path, viridis_colormap)

print("Viridis colormap image saved to:", output_path)

# Load the depth map and the corresponding image
depth_map = cv2.imread('/home/kaai/yolov5/runs/detect/exp14/sample/depth/viridis_colormap.png')
image = cv2.imread('/home/kaai/yolov5/runs/detect/exp14/sample/images/000000.png')

# Resize depth map to match the size of the image
depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

# Adjust brightness and contrast of the depth map
alpha = 1.5  # Contrast control (1.0-3.0)
beta = 20    # Brightness control (0-100)
depth_map_resized = cv2.convertScaleAbs(depth_map_resized, alpha=alpha, beta=beta)

# Create an overlay by adding the depth map on top of the image
overlay = cv2.addWeighted(image, 0.7, depth_map_resized, 0.6, 0)  # Adjust weights to make depth map more visible

# Save the overlay image
cv2.imwrite('/home/kaai/yolov5/runs/detect/exp14/sample/depth/overlay_image.png', overlay)

print("Overlay image saved to:", '/home/kaai/yolov5/runs/detect/exp14/sample/depth/overlay_image.png')
