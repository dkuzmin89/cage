import numpy as np
import cv2
import matplotlib.pyplot as plt
# import fileinput
# import glob
# import sys


def read_img (path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (0, 0), img, 0.3, 0.3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray, img_rgb

def blurred (img, sigmaX=3, sigmaY=3):
    img_bl_gray = cv2.GaussianBlur(img, (sigmaX, sigmaY), 0)
    return img_bl_gray

def edges (img, thr1=120, thr2=255):
    edges = cv2.Canny(img, thr1, thr2)
    return edges

def contour_detection (img, thr1=190, thr2=255):
    zeros = np.zeros(img.shape)
    ret, thresh = cv2.threshold(img, thr1, thr2, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont = cv2.drawContours(zeros, contours, -1, (255, 255, 255), cv2.FILLED)
    return cont, thresh

def merge (img1, img2): ## edges and thresh
    img_comb = cv2.addWeighted(img1, 1, img2, 1, 0)
    return img_comb

def morphologic (img_thr, k_d1=10, k_d2=10, k_o1=10, k_o2=10, iter=2):
    dilation = cv2.dilate(img_thr, np.ones((k_d1,k_d2),np.uint8), iterations=iter)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, np.ones((k_o1,k_o2),np.uint8))
    inversion = cv2.bitwise_not(opening)
    return inversion

def masking (img_rgb):
    n_bg = np.full(img.shape, 255, dtype=np.uint8)
    mask = cv2.bitwise_or(n_bg, n_bg, mask=inversion)
    final_img = cv2.bitwise_and(img_rgb, mask)
    return final_img, n_bg, mask

def put_mask (path1, mask):
    img1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
    img1 = cv2.resize(img1, (0, 0), img, 0.3, 0.3)
    img_rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    masked_img = cv2.bitwise_and(img_rgb1, mask)
    cv2.imwrite('MASKed_img.png', masked_img)
    return img1, img_rgb1, masked_img
    # pass



if __name__ == '__main__':

    path = '5-4.png'
    img, img_gray, img_rgb = read_img(path)
    img_bl_gray = blurred(img_gray)
    edges = edges(img_gray)
    cont, thresh = contour_detection(img_bl_gray)
    # img_comb = merge(edges,thresh)
    inversion = morphologic(thresh)
    final_img, n_bg, mask = masking(img_rgb)

    path1 = '5-1.png'
    img1, img_rgb1, masked_img = put_mask(path1, mask)



    ########
    ##SHOW##
    ########
    plt.imshow(mask, cmap='gray')
    plt.show()
    plt.imshow(final_img)
    plt.show()
    plt.imshow(masked_img)
    plt.show()

    #########
    ##WRITE##
    #########
    cv2.imwrite('MASK.png', mask)



