import os
import shutil

# Change these paths
source_dir = './data/imageNetTest/test'
destination_dir = './data/classifier/test/ImageNet/test'

# Create destination folder if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# List all image files (you can adjust extensions if needed)
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(image_extensions)]

# Sort by modification time (or use key=lambda x: x for name sort)
image_files.sort(key=lambda x: os.path.getmtime(os.path.join(source_dir, x)))

# Get the last 1000 images
last_1000 = image_files[-2000:-1000]

# Copy to destination
for file_name in last_1000:
    src_path = os.path.join(source_dir, file_name)
    dst_path = os.path.join(destination_dir, file_name)
    shutil.copy2(src_path, dst_path)

print(f"Copied {len(last_1000)} images to '{destination_dir}'")
