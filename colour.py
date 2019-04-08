import cv2
import numpy as np
from PIL import Image
import pytesseract
import sys
import os


# get image
fp = sys.argv[1]
rawimg = cv2.imread(fp)
#xyzImg=cv2.cvtColor(rawimg,cv2.COLOR_RGB2XYZ)
labImg=cv2.cvtColor(rawimg,cv2.COLOR_BGR2LAB)
L,A,B = labImg[398][429]
#showimg("xyz", labImg)

print(L*100/255)
print(A - 128)
print(B - 128)
