from  PIL import Image
import os 
from glob import glob

input_folder = "Mazda-2000"
output_folder = "dataset"

os.makedirs(output_folder, exist_ok=True)
 
for i, Img_path in enumerate(glob(os.path.join(input_folder, "*"))):
    try:
        img= Image.open(Img_path).convert("RGB")
        img= img.resize((128, 128))
        save_paht= os.path.join(output_folder, f"img_{i+1}.jpg")
        img.save(save_paht)
    except Exception as e:
        print(f"Error processing {Img_path}: {e}")