import albumentations as A
import cv2
import os

transform = A.Compose([
    A.MotionBlur(blur_limit=3,always_apply=True),
    A.Rotate(limit=1,always_apply=True),
    A.RandomBrightnessContrast(brightness_limit=0.3,p=1)
])

dir = "imgs/train"
masks = "masks"
batch = 1

for file in os.listdir(dir):
    if file.endswith("png"):
        filename = file.split(".")[0]

        img = os.path.join(dir,file)
        mask = os.path.join(masks, filename + "_mask.png")
        if os.path.isfile(mask):
            image = cv2.imread(img)
            mask = cv2.imread(mask)

            for i in range(batch):
                transformed = transform(image=image, mask=mask)
                transformed_image = cv2.cvtColor(transformed['image'],cv2.COLOR_BGR2GRAY)
                transformed_mask = cv2.cvtColor(transformed['mask'],cv2.COLOR_BGR2GRAY)

                cv2.imwrite(os.path.join(dir, filename + "_aumgmented" + str(i) + ".png"), transformed_image)
                cv2.imwrite(os.path.join(masks, filename + "_aumgmented" + str(i) + "_mask.png"), transformed_mask)
