
# import opencv
import cv2
import os 

bgr_path = "additional_images"
output_path = "additional_images_gray"

for img in os.listdir(bgr_path):
    print("converting, ",img)
    f_name = img.split(".")[0] + ".jpg"
    image = cv2.imread(os.path.join(bgr_path,img))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_path,f_name), gray_image)
