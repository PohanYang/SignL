import cv2
import numpy
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

img = cv2.imread('rgbdata/pic337.png')

b, g, r = cv2.split(img)
red = cv2.equalizeHist(r)
green = cv2.equalizeHist(g)
blue = cv2.equalizeHist(b)
img_output = cv2.merge((blue, green, red))
#img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', img_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
