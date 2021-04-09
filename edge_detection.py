import os
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

title_window = 'Detetar a sombra'
data_dir = os.path.join(os.path.dirname(__file__), 'data')
img_name = 'art_shadow.png'
low_threshold = high_threshold = 200
aperture = 3
dilate = 5
erode = 5


img = cv.imread(os.path.join(data_dir, img_name), cv.IMREAD_GRAYSCALE)

res = np.zeros_like(np.hstack((img, img, img)))


def display():
    cv.imshow(title_window, res)


def apply():
    global res
    canny = cv.Canny(img, low_threshold, high_threshold, apertureSize=aperture)
    dil = cv.morphologyEx(canny, cv.MORPH_DILATE, np.ones((dilate, dilate)))
    ero = cv.morphologyEx(dil, cv.MORPH_ERODE, np.ones((erode, erode)))
    res = np.hstack((canny, dil, ero))


def on_low_threshold(v):
    global low_threshold
    low_threshold = v


def on_high_threshold(v):
    global high_threshold
    high_threshold = v


def on_dilate(v):
    global dilate
    dilate = v


def on_erode(v):
    global erode
    erode = v


threshold_max = 256
morph_max = 21

cv.namedWindow(title_window)
cv.createTrackbar('Low Threshold', title_window,
                  low_threshold, threshold_max, on_low_threshold)
cv.createTrackbar('High Threshold', title_window,
                  high_threshold, threshold_max, on_high_threshold)
cv.createTrackbar('Dilate', title_window,
                  dilate, morph_max, on_dilate)
cv.createTrackbar('Erode', title_window,
                  erode, morph_max, on_erode)


while True:
    apply()
    display()
    key = cv.waitKey(100)
    if key == ord('q'):
        break

cv.destroyAllWindows()

# src = img
# dst = morph

# # Copy edges to the images that will display the results in BGR
# cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
# cdstP = np.copy(cdst)

# lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#         cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

# linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

# if linesP is not None:
#     for i in range(0, len(linesP)):
#         l = linesP[i][0]
#         cv.line(cdstP, (l[0], l[1]), (l[2], l[3]),
#                 (0, 0, 255), 3, cv.LINE_AA)

# cv.imshow("Source", src)
# cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
# cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

# cv.waitKey()
