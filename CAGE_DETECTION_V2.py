import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def applyMask(image, mask):
    image_size = (image.shape[1], image.shape[0])
    tmp_mask = cv2.resize(mask, image_size)
    return cv2.bitwise_and(image,image, mask=tmp_mask)

size = (1280, 760)

input_dir = 'input'

input_images = [cv2.imread(os.path.join(input_dir, i)) for i in sorted(os.listdir(input_dir)) if i.endswith('png')]

for i in range(len(input_images)):
    input_images[i] = cv2.resize(input_images[i], size)

common = input_images[0].copy()

for i in range(len(input_images)):
    common = cv2.addWeighted(common, 0.5, input_images[i], 0.5, 0)

common = cv2.resize(common, (0, 0), common, 0.1, 0.1)

gray = cv2.cvtColor(common, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 110, 400)

ret, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)

img_comb = edges + thresh

dilation = cv2.dilate(img_comb, np.ones((3,3),np.uint8), iterations=3)
erosion = cv2.erode(dilation, np.ones((2,2),np.uint8), iterations=2)

mask = cv2.bitwise_not(erosion)
mask = cv2.resize(mask, size)

mask = cv2.GaussianBlur(mask, (31,31), 0)
mask = cv2.GaussianBlur(mask, (21,21), 0)

folder_to_apply = "pics"
folder_with_result = "output"
if not os.path.exists(folder_with_result):
    os.makedirs(folder_with_result)

for pic in os.listdir(folder_to_apply):
    if pic.endswith('png'):
        pic_name = pic.rsplit('.', 1)[0]
        img = cv2.imread(os.path.join(folder_to_apply, pic))
        result = applyMask(img, mask)
        cv2.imwrite(os.path.join(folder_with_result, pic_name + '_masked.png'), result)


plt.imshow(mask, cmap='gray')
plt.show()

