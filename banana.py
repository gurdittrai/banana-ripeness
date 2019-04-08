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
    
def getcontours(rawimg, img, techname):
    #copies
    imgcanny = rawimg.copy()
    test = rawimg.copy()

    # get contour CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_L1
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    lCon = None
    lArea = 0
    for contour in contours:
        epsilion = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilion, True)
        
        area = cv2.contourArea(contour)
        # if area > 100 :
        #     count = count + 1
        #     test = cv2.drawContours(test, [approx], 0, (255,0,0), 3)

        if area > lArea:
            lArea = area
            lCon = contour
        
        imgcanny = cv2.drawContours(imgcanny, [approx], 0, (255,0,0), 3)

    #largest
    epsilion = 0.01 * cv2.arcLength(lCon, True)
    approx = cv2.approxPolyDP(lCon, epsilion, True)
    test = cv2.drawContours(test, [approx], 0, (255,0,0), 3)

    #name
    cannyname = "%s: %d" % (techname, len(contours))
    testname = "%s Largest Area" % (techname)

    #show image
    showimg(cannyname, imgcanny)
    showimg(testname, test)
        

# get image
fp = sys.argv[1]
rawimg = cv2.imread(fp)


##########################
# background subtraction #
##########################

#convert grayscale
grayimg = cv2.cvtColor(rawimg, cv2.COLOR_RGB2GRAY)

#improve contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
imgCLAHE = clahe.apply(grayimg)

# showimg("grayimg", grayimg)
# showimg("CLAHE", imgCLAHE)

#remove background
nowhite = rmvWhiteBackground(rawimg)
nowhite = cv2.cvtColor(nowhite, cv2.COLOR_RGB2GRAY)
nowhite = clahe.apply(nowhite)

#blur to remove small details
blurimg = cv2.GaussianBlur(nowhite, (5, 5), 0)
# showimg("nowhite", nowhite)

##################
# edge detection #
##################

# #using canny
# canny = cv2.Canny(blurimg, 100, 150)
# # showimg("Canny", canny)

# # Perform morphology MORPH_CLOSE MORPH_TOPHAT
# se = np.ones((7,7), dtype='uint8')
# image_close = cv2.morphologyEx(canny, cv2.MORPH_TOPHAT, se)
# # showimg("image_close using canny", image_close)

# getcontours(rawimg, canny, "Canny")

##################
# edge detection #
##################

#using laplacian
ddepth = 3
kernel_size = 3
imglap = cv2.Laplacian(blurimg, ddepth=cv2.CV_8U,ksize = 3)
# showimg("imglap", imglap)

# Perform morphology MORPH_CLOSE MORPH_TOPHAT
se = np.ones((7,7), dtype='uint8')
image_close_lap = cv2.morphologyEx(imglap, cv2.MORPH_TOPHAT, se)
# showimg("image_close using laplacian", image_close_lap)

getcontours(rawimg, image_close_lap, "Laplacian")

##################
# edge detection #
##################

# using hough circles
circles = cv2.HoughCircles(blurimg,cv2.HOUGH_GRADIENT,1,120,param1=50,param2=30,minRadius=50, maxRadius=150)

# wait for user to exit
cv2.waitKey(0) 
cv2.destroyAllWindows()
