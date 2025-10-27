# import os

# # Base path
# base_path = "dataset_fixed/train"

# # 30 folder names
# folders = [
#     "aqidi", "bale", "brozovic", "bruyne", "cancelo", "casimiro", "costa", "cummins",
#     "deligt", "dias", "fernandes", "gordon", "haaland", "kohli", "mane", "marcelo",
#     "mbappe", "mendes", "messi", "ozil", "pepe", "pickford", "ramos", "rodriguez",
#     "ronaldo", "saka", "shelton", "sinner", "trent", "vitinha"
# ]

# # Create folders and dummy image files
# for folder in folders:
#     folder_path = os.path.join(base_path, folder)
#     os.makedirs(folder_path, exist_ok=True)  # Create folder if not exists

#     # Create dummy image files
#     for i in range(1, 10):
#         file_path = os.path.join(folder_path, f"{i}.jpg")
#         with open(file_path, "w") as f:
#             f.write("")  # create empty file

# print("Dataset folder structure created successfully!")




import os
from PIL import Image
import numpy as np

# -------------------------------
# Parameters
# -------------------------------
source_path = "train"           # মূল train folder
dest_path = "face recognition/6_Classification & Matching/fixed_train"  # processed images save folder
image_size = (224, 224)                 # reshape size

os.makedirs(dest_path, exist_ok=True)   # নতুন folder create

# -------------------------------
# Process each folder
# -------------------------------
person_folders = os.listdir(source_path)
person_folders.sort()  # optional

for folder_name in person_folders:
    src_folder = os.path.join(source_path, folder_name)
    dst_folder = os.path.join(dest_path, folder_name)
    os.makedirs(dst_folder, exist_ok=True)  # create folder in new location
    
    for file in os.listdir(src_folder):
        src_file = os.path.join(src_folder, file)
        try:
            img = Image.open(src_file).convert('RGB')  # সব image RGB তে convert
            img = img.resize(image_size)               # reshape
            
            # Save as JPG
            filename = os.path.splitext(file)[0] + ".jpg"  # convert name to .jpg
            dst_file = os.path.join(dst_folder, filename)
            img.save(dst_file, "JPEG")
            
        except Exception as e:
            print(f"Error processing {src_file}: {e}")

print("All images converted, resized, and saved in:", dest_path)