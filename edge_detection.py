import os
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

title_window = 'Detetar a sombra'
low_threshold = 134
high_threshold = 155
aperture = 3
dilate = 10
erode = 18


img = cv.imread(
    'stress2.png', cv.IMREAD_GRAYSCALE)

canny = np.zeros_like(img)
morph = np.zeros_like(img)
filtered_canny = np.zeros_like(img)
dilate_up = np.zeros_like(img)


def display():
    cv.imshow(title_window, canny)
    cv.imshow('after morph', morph)
    cv.imshow('filter canny', filtered_canny)
    cv.imshow('dilate up', dilate_up)


def apply():
    global canny, morph, dilate_up, filtered_canny
    start = img
    # ret, start = cv.threshold(img, low_threshold, 255, cv.THRESH_BINARY_INV)

    canny = cv.Canny(start, low_threshold, high_threshold,
                     apertureSize=aperture)

    dil = cv.morphologyEx(canny, cv.MORPH_DILATE, np.ones((dilate, dilate)))
    morph = cv.morphologyEx(dil, cv.MORPH_ERODE, np.ones((erode, erode)))

    kernel = np.zeros((dilate + 5, dilate+5))
    kernel[int(dilate+5/2):, :] = np.ones((dilate+5-int(dilate+5/2), dilate+5))
    kernel = kernel.astype(np.uint8)
    dilate_up = cv.morphologyEx(morph, cv.MORPH_DILATE, kernel)
    #print(dilate_up[(dilate_up != 0) & (dilate_up != 255)])
    filtered_canny = np.zeros_like(img, dtype=np.uint8)
    filtered_canny[(dilate_up == 255) & (canny == 255)] = 255
    #print(((dilate_up == 255) & (canny == 255)).any())


def on_low_threshold(v):
    global low_threshold
    low_threshold = v

def on_aperture(v):
    global aperture
    aperture = v + 1 - v%2

def on_high_threshold(v):
    global high_threshold
    high_threshold = v


def on_dilate(v):
    global dilate
    dilate = v


def on_erode(v):
    global erode
    erode = v


morph_max = 31
threshold_max = 300

cv.namedWindow(title_window)
cv.createTrackbar('Low Threshold', title_window,
                  low_threshold, threshold_max, on_low_threshold)
cv.createTrackbar('High Threshold', title_window,
                  high_threshold, threshold_max, on_high_threshold)
cv.createTrackbar('Dilate', title_window,
                  dilate, morph_max, on_dilate)
cv.createTrackbar('Erode', title_window,
                  erode, morph_max, on_erode)
cv.createTrackbar('Aperture', title_window,
                  aperture, 7, on_aperture)


while True:
    apply()
    display()
    key = cv.waitKey(100)
    if key == ord('q'):
        break

cv.destroyAllWindows()
cv.imwrite('result.png', morph)
cv.imwrite('result_new.png', filtered_canny)

# kernel_dilate = cv.getStructuringElement(
#     cv.MORPH_ELLIPSE, (dilate, dilate))

# print(kernel_dilate)
# kernel_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erode, erode))
