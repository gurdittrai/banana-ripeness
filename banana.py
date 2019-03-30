# Reading Car Plate Numbers
import cv2
import numpy as np
from PIL import Image
import pytesseract
import sys
import os

def showimg(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)

# get image
fp = sys.argv[1]
rawimg = cv2.imread(fp)

# crop
h = np.size(rawimg, 0)
w = np.size(rawimg, 1)

x = (int)(w * 0.04)
y = (int)(h * 0.22)
w = (int)(w - x)
h = (int)(h - y)
cropimg = rawimg[y:h, x:w]

# save offset
rawoffset = (x,y)

# convert grayscale
grayimg = cv2.cvtColor(cropimg, cv2.COLOR_RGB2GRAY)
if grayimg is None:
    print("error opening file\n")

# seperate background
cv2.imwrite("gray.png", grayimg)

thresh = 150
maxval = 255
th, dst = cv2.threshold(grayimg, thresh, maxval, cv2.THRESH_BINARY)

# showimg("raw", rawimg)
# showimg("binary", dst)

#save to file
fname = "binary.png"
cv2.imwrite(fname, dst)

#edge
blurimg = cv2.GaussianBlur(grayimg, (11, 11), 0)
lp = cv2.Laplacian(blurimg, cv2.CV_64F, ksize=5)
canny = cv2.Canny(grayimg, 100, 150)

showimg("LaPlace", lp)
showimg("Canny", canny)

# get contour
_, contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_list = []
for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    # if area > 500 :
        # cv2.rectangle(cropimg, (x, y), (x + w, y + h), (255, 0, 255), 2)

    if area > 200 :
        contour_list.append(contour)

cv2.drawContours(rawimg, contour_list,  -1, (0,180,255), 2, offset=rawoffset)
showimg("Objects Detected Contour", rawimg)

# wait for user to exit
cv2.waitKey(0) 
cv2.destroyAllWindows()
