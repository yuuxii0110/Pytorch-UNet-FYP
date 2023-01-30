import os
import cv2
import numpy as np

dir = "test_imgs/"
out = "painted_imgs_all"

for file in os.listdir(dir):
    if not file.endswith("_predicted_mask.jpg") and os.path.isfile(os.path.join(dir,file)) and (file.endswith("jpg") or file.endswith("png")):
        print("converting, ", file)
        f_img = file
        f_mask = f_img.split(".")[0] + "_predicted_mask.jpg"
        img = cv2.imread(os.path.join(dir,f_img))
        mask = cv2.imread(os.path.join(dir,f_mask),1)
        # i, j = np.where(mask > limit)
        # for i in mask:
        #     for m in i:
        #         if m[0] == 127 and m[1] == 127 and m[2] == 127:
        #             m[0] = 0
        #             m[1] = 0
        #             m[2] = 255

        dst = cv2.addWeighted(img, 0.65, mask, 0.35, 0)
        cv2.imwrite(os.path.join(out,file),dst)


