# Reading Car Plate Numbers
import cv2
import numpy as np
from PIL import Image
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

def rmvBackground_minor(img):
    #use this if loading in bgr
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img
    #image hsv
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

    #detect colour space
    lower_white = np.array([0,100,120])
    upper_white = np.array([50,255,255])

    #mask
    mask = cv2.inRange(img_hsv, lower_white, upper_white)

    #mask on rgb img
    banana = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    return banana
    
            

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

def rmvBackground(img):
    #image rgb
    img_rgb = img

    #image hsv
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

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
    # Y_banana = cv2.bitwise_and(img_rgb, img_rgb, mask=yellow_mask)
    # Y_background = cv2.bitwise_and(img_rgb, img_rgb, mask=yellow_mask_inv)
    # G_banana = cv2.bitwise_and(img_rgb, img_rgb, mask=green_mask)
    # G_background = cv2.bitwise_and(img_rgb, img_rgb, mask=green_mask_inv)
    # B_banana = cv2.bitwise_and(img_rgb, img_rgb, mask=brown_mask)
    # B_background = cv2.bitwise_and(img_rgb, img_rgb, mask=brown_mask_inv)

    #Combine masks into one picture to get to total banana
    banana = cv2.bitwise_and(img_hsv, img_hsv, mask=green_mask+yellow_mask+brown_mask)
    background = cv2.bitwise_and(img_hsv, img_hsv, mask=brown_mask_inv+yellow_mask_inv+green_mask_inv)

    #showimg('background', background)
    #showimg('banana', banana)

    #convert the picure thie the background removed back to rgb for colour space LAB analysis
    banana = cv2.cvtColor(banana, cv2.COLOR_HSV2BGR)
    return banana
    showimg('banana', banana)
            
    
def getcontours(rawimg, img, techname):
    #copies
    mask = np.zeros_like(rawimg, dtype=np.uint8)

    #get contour CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_L1
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lCon = None
    lArea = 0
    for contour in contours:
        #contour parameters
        epsilion = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilion, True)
        
        #get area
        area = cv2.contourArea(contour)

        #get largest
        if area > lArea:
            lArea = area
            lCon = contour

    #largest contour area
    epsilion = 0.01 * cv2.arcLength(lCon, True)
    approx = cv2.approxPolyDP(lCon, epsilion, True)
    mask = cv2.drawContours(mask, [approx], 0, (255,255,255), 3) #, cv2.FILLED)

    #close shape
    cv2.fillPoly(mask, pts = [approx], color=(255,255,255))

    #mask
    mask = cv2.bitwise_not(mask)

    return mask

    
        
###################
# obtaining image #
###################

# get image
fp = sys.argv[1]
rawimg = cv2.imread(fp)


##########################
# background subtraction #
##########################

#remove background
noback = rmvBackground_minor(rawimg)

#convert grayscale
grayimg = cv2.cvtColor(rawimg, cv2.COLOR_RGB2GRAY)

#improve contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
imgCLAHE = clahe.apply(grayimg)

#grayscale
nobackgray = cv2.cvtColor(noback, cv2.COLOR_RGB2GRAY)
nobackgray = clahe.apply(nobackgray)

#blur to remove small details
blurimg = cv2.GaussianBlur(nobackgray, (5, 5), 0)

##################
# edge detection #
##################

#using laplacian
ddepth = 3
kernel_size = 3
imglap = cv2.Laplacian(blurimg, ddepth=cv2.CV_8U,ksize = 3)

#perform morphology to close gaps  
se = np.ones((7,7), dtype='uint8')
image_close_lap = cv2.morphologyEx(imglap, cv2.MORPH_TOPHAT, se) # (MORPH_CLOSE MORPH_TOPHAT)

#getting contours
mask = getcontours(rawimg, image_close_lap, "Laplacian")

#apply mask
banana_area = rmvBackground(rawimg)
cv2.normalize(rawimg, banana_area, dtype=cv2.CV_8UC1 ,mask=mask)

showimg("banana_area", banana_area)

###################
# judging ripness #
###################

#calculate the L*A*B* values
# LABConversion(banana_area)


# wait for user to exit
cv2.waitKey(0) 
cv2.destroyAllWindows()
