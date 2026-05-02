import os

folder_path = "/data/dataset_1/train/scissors"

files = sorted(os.listdir(folder_path))

image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for i, filename in enumerate(image_files, start=1):
    new_name = f"scissors_{i:03d}.jpg"
    
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    
    os.rename(old_path, new_path)

print("Renaming complete.")