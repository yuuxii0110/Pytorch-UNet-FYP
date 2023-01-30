from PIL import Image
import os
dir = "masks"
out = "masks_gif"

for file in os.listdir(dir):
    savefile_name = file.split(".")[0] + ".gif"
    img = Image.open(os.path.join(dir,file))
    print(file)
    img.save(os.path.join(out,savefile_name)); 
