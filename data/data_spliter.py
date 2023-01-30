import os
import shutil

data_folder = "storage"
#color_folder = "additional_images"
target_folder = "imgs"
#color_target_folder = "additional_images_gray/labeled/imgs_raw"
#other = "additional_images_gray/not_labeled"

for file in os.listdir(data_folder):
    if file.endswith(".png"):
        f_idx = file.split(".")[0]
        f_name = f_idx + ".json"
        if os.path.exists(os.path.join(data_folder,f_name)):
            src = os.path.join(data_folder,file)
            dst = os.path.join(target_folder,file)
            #src2 = os.path.join(color_folder,f_idx+".png")
            #dst2 = os.path.join(color_target_folder,f_idx+".png")
            shutil.copyfile(src, dst)
            #shutil.copyfile(src2, dst2)
            print(os.path.join(data_folder,f_name), "found")
        else:
            print(os.path.join(data_folder,f_name), "not found")
        # else:
        #     src = os.path.join(data_folder,file)
        #     dst = os.path.join(other,file)
        #     shutil.copyfile(src, dst)    
