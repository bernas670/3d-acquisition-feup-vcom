import os
import sys
import math
import cv2 as cv
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression, RANSACRegressor

def calculatePlaneCoefs(points):
  x = points[:, :1]
  y = points[:, 1]

  # estimate Ax + B = y (C = 1)
  # validateData ensures that points used to estimate the plane are in different Z planes
  # without this the algorithm was considering all points in other planes as outliers
  reg = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=True),
    #residual_threshold=0.1,
    #max_trials=1000
    ).fit(x, y)

  return reg.estimator_.coef_, reg.estimator_.intercept_, reg.inlier_mask_



def main(argv):
    img_name = "/home/luispcunha/repos/feup/vcom/vcom-proj1/data/2ndtry.png"
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Loads an image
    src = cv.imread(
        '/home/luispcunha/repos/feup/vcom/vcom-proj1/result_new.png', cv.IMREAD_GRAYSCALE)

    # Check if image is loaded fine'
    if src is None:
        print('Error opening image!')
        return -1

    # pronto, david, tás a ver?
    points = cv.findNonZero(src)

    _, _, inliers = calculatePlaneCoefs(np.reshape(points, (-1, 2)))

    bottom = np.zeros_like(src, dtype=np.uint8)
    for pt in points[inliers]:
        bottom[pt[0,1], pt[0,0]] = 255

    points_top = points[np.logical_not(inliers)]

    _, _, inliers = calculatePlaneCoefs(np.reshape(points_top, (-1, 2)))
    top = np.zeros_like(src, dtype=np.uint8)
    for pt in points_top[inliers]:
        top[pt[0, 1], pt[0, 0]] = 255

    cv.imshow("Source", src)
    cv.imshow("Bottom", bottom)
    cv.imshow("Top", top)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
