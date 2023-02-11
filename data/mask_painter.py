import os
import cv2
import numpy as np

dir = "b3_outdoor_test_imgs/"
out = "b3_outdoor_test_imgs/painted_imgs"

for file in os.listdir(dir):
    if not file.endswith("_predicted_mask.jpg") and os.path.isfile(os.path.join(dir,file)) and (file.endswith("jpg") or file.endswith("png")):
        print("converting, ", file)
        f_img = file
        f_mask = f_img.split(".")[0] + "_predicted_mask.jpg"
        img = cv2.imread(os.path.join(dir,f_img))
        mask = cv2.imread(os.path.join(dir,f_mask),1)
        indices = np.where(mask>0)
        img_copy = img.copy()
        img_copy[indices[0],indices[1],:] = (128,0,128)
        dst = cv2.addWeighted(img, 0.65, img_copy, 0.35, 0)
        cv2.imwrite(os.path.join(out,file),dst)


