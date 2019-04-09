# Banana ripeness algoirthm
# By: Andrew Maklingham, Gurditt Rai
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

        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(roi, roi, mask=mask)
        fg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        th, dst = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)

        cv2.imshow('bipanel', bipanel)
        showimg('binary', dst)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break

    #save to file
    fname = "binary.png"
    cv2.imwrite(fname, dst)

def LABConversion(img):
    height, width = img.shape[:2]
    labImg=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    green = 0.0
    yellow = 0.0
    brown = 0.0

    for i in range(height):
        for k in range(width):
            L,a,b = labImg[i][k]
            a = a - 128
            b = b - 128
            L = L * 100 / 225
            if L != 100 and L != 0:
                if a < 18 and b > 47 and a > -17:
                    yellow += 1
                elif a < -7:
                    green += 1
                elif a > 10 and b < 47 and L > 19:
                    brown += 1

    total = green + yellow + brown
    greenPer = (green/total)*100
    yellowPer = (yellow/total)*100
    brownPer = (brown/total)*100
    print("green: ", greenPer)
    print("yellow: ", yellowPer)
    print("brown: ", brownPer)
    if brownPer >= 35:
        if brownPer >= 50:
            print("banana is very over ripe")
        else:
            print("banana is over ripe")
    elif greenPer >= 30:
        if greenPer >= 60:
            print("banana is very unripe")
        else:
            print("banana is unripe")
    elif yellowPer > 50:
        print("banana is ripe")
    else:
        print("banana is over ripe")

def rmvWhiteBackground(img):
    #use this if loading in bgr
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_rgb = img
    #image hsv
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

    while True:
        #colour values for green
        G_lower = np.array([28,46,45])
        G_upper = np.array([70,255,255])

        #green mask
        green_mask = cv2.inRange(img_hsv, G_lower, G_upper)
        green_mask_inv = cv2.bitwise_not(green_mask)

        #yellow values
        Y_lower = np.array([18,85,0])
        Y_upper = np.array([28,255,255])

        #brown values
        B_lower = np.array([2,20,20])
        B_upper = np.array([12,255,150])

        #yellow mask
        yellow_mask = cv2.inRange(img_hsv, Y_lower, Y_upper)
        yellow_mask_inv = cv2.bitwise_not(yellow_mask)

        #brown mask
        brown_mask = cv2.inRange(img_hsv, B_lower, B_upper)
        brown_mask_inv = cv2.bitwise_not(brown_mask)

        #mask on rgb img. Each of these are only used for testing purposes.
        Y_banana = cv2.bitwise_and(img_rgb, img_rgb, mask=yellow_mask)
        Y_background = cv2.bitwise_and(img_rgb, img_rgb, mask=yellow_mask_inv)
        G_banana = cv2.bitwise_and(img_rgb, img_rgb, mask=green_mask)
        G_background = cv2.bitwise_and(img_rgb, img_rgb, mask=green_mask_inv)
        B_banana = cv2.bitwise_and(img_rgb, img_rgb, mask=brown_mask)
        B_background = cv2.bitwise_and(img_rgb, img_rgb, mask=brown_mask_inv)

        #Combine masks into one picture to get to total banana
        banana = cv2.bitwise_and(img_hsv, img_hsv, mask=green_mask+yellow_mask+brown_mask)
        background = cv2.bitwise_and(img_hsv, img_hsv, mask=brown_mask_inv+yellow_mask_inv+green_mask_inv)

        #showimg('background', background)
        #showimg('banana', banana)

        #convert the picure thie the background removed back to rgb for colour space LAB analysis
        banana = cv2.cvtColor(banana, cv2.COLOR_HSV2BGR)
        showimg('banana', banana)

        #exit on escape
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            #calculate the L*A*B* values
            LABConversion(banana)
            H, S, V = cv2.split(banana)
            return V

# get image
fp = sys.argv[1]
rawimg = cv2.imread(fp)
#remove background
noBack = rmvWhiteBackground(rawimg)

def edge(img):

    #improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    imgCLAHE = clahe.apply(img)
    # showimg("grayimg", grayimg)
    # showimg("CLAHE", imgCLAHE)

    #edge detection
    #using canny
    blurimg = cv2.GaussianBlur(imgCLAHE, (5, 5), 0)
    canny = cv2.Canny(blurimg, 100, 150)
    #showimg("Canny", canny)

    # Perform morphology
    se = np.ones((7,7), dtype='uint8')
    image_close = cv2.morphologyEx(canny, cv2.MORPH_TOPHAT, se)
    ##showimg("image_close", image_close)

    #copies
    imgcanny = img.copy()
    test = img.copy()

    # get contour CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_L1
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for contour in contours:
        epsilion = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilion, True)

        area = cv2.contourArea(contour)
        if area > 100 :
            count = count + 1
            testcnt = cv2.drawContours(test, [approx], 0, (255,0,0), 3)
        img = cv2.drawContours(imgcanny, [approx], 0, (255,0,0), 3)

    cannyname = "Objects Detected Canny: %d" % (len(contours))
    testname = "Objects Detected Test: %d" % (count)
    #showimg(cannyname, imgcanny)

edge(noBack)

# wait for user to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
