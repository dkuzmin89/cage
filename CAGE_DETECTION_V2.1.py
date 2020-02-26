import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#readind files, starting from 1st; then resizind to 1280:760.
def img_read(input_dir, size):
    input_images = [cv2.imread(os.path.join(input_dir, i)) for i in sorted(os.listdir(input_dir)) if i.endswith('png')]

    for i in range(len(input_images)):
        input_images[i] = cv2.resize(input_images[i], size)

    common = input_images[0].copy()

    for i in range(len(input_images)):
        common = cv2.addWeighted(common, 0.5, input_images[i], 0.5, 0)

    img = cv2.resize(common, (0, 0), common, 0.1, 0.1)
    return img


def processing(img, c1=110, c2=400):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, c1, c2)
    _, thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    img_comb = edges + thresh
    return img_comb


def morphologic(img_comb, iter_d=3, iter_e=1):
    dilation = cv2.dilate(img_comb, np.ones((3, 3), np.uint8), iterations=iter_d)
    img_mask = cv2.erode(dilation, np.ones((2, 2), np.uint8), iterations=iter_e)
    return img_mask


def mask(img_mask):
    mask = cv2.bitwise_not(img_mask)
    mask = cv2.resize(mask, size)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


def apply_mask(image, mask):
    image_size = (image.shape[1], image.shape[0])
    tmp_mask = cv2.resize(mask, image_size)
    return cv2.bitwise_and(image, image, mask=tmp_mask)


if __name__ == '__main__':

    size = (1280, 760)
    input_dir = 'input'
    folder_to_apply = "pics"
    folder_with_result = "output"

    img = img_read(input_dir, size)
    img_comb = processing(img)
    erosion = morphologic(img_comb)
    mask = mask(erosion)

    cv2.imwrite('output/MASK.png', mask)


    if not os.path.exists(folder_with_result):
        os.makedirs(folder_with_result)

    for pic in os.listdir(folder_to_apply):
        if pic.endswith('png'):
            pic_name = pic.rsplit('.', 1)[0]
            img = cv2.imread(os.path.join(folder_to_apply, pic))
            result = apply_mask(img, mask)
            cv2.imwrite(os.path.join(folder_with_result, pic_name + '_masked.png'), result)

plt.imshow(mask, cmap='gray')
plt.show()
plt.imshow(result)
plt.show()
