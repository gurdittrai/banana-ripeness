# Reading Car Plate Numbers
import cv2
import numpy as np
from PIL import Image
import pytesseract
import sys
import os

def showimg(name, img):
    #size
    h = np.size(img, 0)
    w = np.size(img, 1)
    
    #fit to window
    aratio = w/float(h)
    h = 700
    w = int(h * aratio)
    

    #window
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.imshow(name, img)

def nothing(x):
    pass

def binaryimg(img):
    bipanel = np.zeros([100, 700], np.uint8)
    cv2.namedWindow('bipanel')
    cv2.resizeWindow('bipanel', 500, 100)
    cv2.createTrackbar('thr', 'bipanel', 0, 255, nothing)
    cv2.createTrackbar('max', 'bipanel', 0, 255, nothing)

    while True:
        thresh = cv2.getTrackbarPos('thr', 'bipanel')
        maxval = cv2.getTrackbarPos('max', 'bipanel')
    
        # mask = cv2.inRange(hsv, lower_green, upper_green)
        # mask_inv = cv2.bitwise_not(mask)
    
        # bg = cv2.bitwise_and(roi, roi, mask=mask)
        # fg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        th, dst = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
    
        cv2.imshow('bipanel', bipanel)
        showimg('binary', dst)
    
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break

    #save to file
    # fname = "binary.png"
    # cv2.imwrite(fname, dst)

def rmvWhiteBackground(img):
    #image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

    #panel
    hsv_panel = np.zeros([100, 700], np.uint8)
    cv2.namedWindow('hsv_panel')
    # cv2.resizeWindow('hsv_panel', 500, 100)
    #lower
    cv2.createTrackbar('lh', 'hsv_panel', 0, 255, nothing)
    cv2.createTrackbar('ls', 'hsv_panel', 0, 255, nothing)
    cv2.createTrackbar('lv', 'hsv_panel', 0, 255, nothing)
    #upper
    cv2.createTrackbar('uh', 'hsv_panel', 0, 255, nothing)
    cv2.createTrackbar('us', 'hsv_panel', 0, 255, nothing)
    cv2.createTrackbar('uv', 'hsv_panel', 0, 255, nothing)

    while True:
        lh = cv2.getTrackbarPos('lh', 'hsv_panel')
        ls = cv2.getTrackbarPos('ls', 'hsv_panel')
        lv = cv2.getTrackbarPos('lv', 'hsv_panel')
        uh = cv2.getTrackbarPos('uh', 'hsv_panel')
        us = cv2.getTrackbarPos('us', 'hsv_panel')
        uv = cv2.getTrackbarPos('uv', 'hsv_panel')

        #detect colour space
        lower_white = np.array([lh,ls,lv])
        upper_white = np.array([uh,us,uv])

        #mask
        mask = cv2.inRange(img_hsv, lower_white, upper_white)
        mask_inv = cv2.bitwise_not(mask)

        #mask on rgb img
        background = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        banana = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_inv)
    
        #show image and panel
        cv2.imshow('hsv_panel', hsv_panel)
        showimg('background', background)
        showimg('banana', banana)
    
        #exit on escape
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break

# get image
fp = sys.argv[1]
rawimg = cv2.imread(fp)

# convert grayscale
grayimg = cv2.cvtColor(rawimg, cv2.COLOR_RGB2GRAY)
if grayimg is None:
    print("error opening file\n")

# seperate background
cv2.imwrite("gray.png", grayimg)

#binary
# binaryimg(grayimg)

#remove background
rmvWhiteBackground(rawimg)

#edge
blurimg = cv2.GaussianBlur(grayimg, (11, 11), 0)
lp = cv2.Laplacian(blurimg, cv2.CV_64F, ksize=5)
canny = cv2.Canny(rawimg, 100, 150)

showimg("LaPlace", lp)
showimg("Canny", canny)

# get contour
# _, contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# contour_list = []
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if area > 500 :
#         contour_list.append(contour)

# cv2.drawContours(rawimg, contour_list,  -1, (0,180,255), 2)
# showimg("Objects Detected Contour", rawimg)

# wait for user to exit
cv2.waitKey(0) 
cv2.destroyAllWindows()
