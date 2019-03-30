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
    #use this if loading in bgr
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img
    #image hsv
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

    #panel
    hsv_panel = np.zeros([100, 700], np.uint8)
    cv2.namedWindow('hsv_panel')
    #lower
    cv2.createTrackbar('lh', 'hsv_panel', 0, 255, nothing)
    cv2.createTrackbar('ls', 'hsv_panel', 100, 255, nothing)
    cv2.createTrackbar('lv', 'hsv_panel', 120, 255, nothing)
    #upper
    cv2.createTrackbar('uh', 'hsv_panel', 50, 255, nothing)
    cv2.createTrackbar('us', 'hsv_panel', 255, 255, nothing)
    cv2.createTrackbar('uv', 'hsv_panel', 255, 255, nothing)

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
        banana = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        background = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_inv)
    
        #show image and panel
        cv2.imshow('hsv_panel', hsv_panel)
        showimg('background', background)
        showimg('banana', banana)
    
        #exit on escape
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return banana

        

# get image
fp = sys.argv[1]
rawimg = cv2.imread(fp)



#binary
# binaryimg(grayimg)

#convert grayscale
grayimg = cv2.cvtColor(rawimg, cv2.COLOR_RGB2GRAY)
#improve contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
imgCLAHE = clahe.apply(grayimg)
# showimg("grayimg", grayimg)
# showimg("CLAHE", imgCLAHE)

#remove background
# nowhite = rmvWhiteBackground(rawimg)
# nowhite = cv2.cvtColor(rawimg, cv2.COLOR_RGB2GRAY)
# nowhite = clahe.apply(nowhite)

#edge detection
#using canny
blurimg = cv2.GaussianBlur(imgCLAHE, (5, 5), 0)
canny = cv2.Canny(blurimg, 100, 150)
showimg("Canny", canny)

# get contour
imgcanny = rawimg.copy()
test = rawimg.copy()
_, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1) #cv2.CHAIN_APPROX_SIMPLE)

count = 0
for contour in contours:
    epsilion = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilion, True)
    
    area = cv2.contourArea(contour)
    if area > 100 :
        count = count + 1
        testcnt = cv2.drawContours(test, [approx], 0, (255,0,0), 3)
    img = cv2.drawContours(imgcanny, [approx], 0, (255,0,0), 3)

cannyname = "Objects Detected Canny: %d vs %d " % (len(contours), count)
showimg('Raw', rawimg)
showimg(cannyname, imgcanny)
showimg('test', test)



# wait for user to exit
cv2.waitKey(0) 
cv2.destroyAllWindows()
