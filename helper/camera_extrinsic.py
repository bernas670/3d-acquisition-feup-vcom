import numpy as np
import cv2
import glob
from itertools import count

CAMERA_ID = 'david'
CHESSBOARD_SQUARE_LENGTH_MM = 1
CHESSBOARD_DIMENSIONS = (9,6)
CAPTURE_DEVICE = 0
FRAME_COUNT = 0
WAIT_PERIOD_MS = 5000

'''
INTRINSIC PARAMETERS
'''

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESSBOARD_DIMENSIONS[0] * CHESSBOARD_DIMENSIONS[1],3), np.float32)

# TODO: check if this is the correct way to adjust the sizes. Used this accordint to: https://stackoverflow.com/questions/37310210/camera-calibration-with-opencv-how-to-adjust-chessboard-square-size
objp[:,:2] = np.mgrid[0:CHESSBOARD_DIMENSIONS[0],0:CHESSBOARD_DIMENSIONS[1]].T.reshape(-1,2) * CHESSBOARD_SQUARE_LENGTH_MM

objp[:, [1, 0]] = objp[:, [0, 1]]

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(f'data/calibration/{CAMERA_ID}/intrinsic/*.png')
print(f'Found {len(images)} calibration images')

for fname in images:
  img = cv2.imread(fname)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Find the chess board corners
  ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_DIMENSIONS, None)
  # print(corners)
  # cv2.imshow('image', img)
  # cv2.waitKey()
  # # If found, add object points, image points (after refining them)
  if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)
  else:
    print(f'Could not find chessboard corners in image {fname}')

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(objpoints))
# print('Image Dimensions')
# print(gray.shape)
# print('Intrinsic Matrix')
# print(mtx)
# print('Distortion Coefficients')
# print(dist)
# print('Rotation Vectors')
# print(rvecs)
# print('Translation Vectors')
# print(tvecs)

# Draw axis
def draw(img, corners, imgpts):
  corner = tuple(corners[0].ravel())
  img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
  img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
  img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
  return img

axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

captureDevice = cv2.VideoCapture(CAPTURE_DEVICE)

'''
EXTRINSIC PARAMETERS
'''

images = glob.glob(f'data/calibration/{CAMERA_ID}/13-04-2021b/extrinsic/*.png')
print(f'Found {len(images)} calibration images')

for fname in images:              
  frame = cv2.imread(fname)
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_DIMENSIONS, None)
  
  if ret == True:
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    # Find the rotation and translation vectors.
    ret, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, mtx, dist)
    # ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
    # project 3D points to image plane
    imgpts2, jac = cv2 .projectPoints(axis, rvecs, tvecs, mtx, dist)
    
    imgpts, jact = cv2.projectPoints(np.float32([[2,2,0], [8,-2,0], [-2,0,0], [0, 15, 0]]).reshape(-1,3), rvecs, tvecs, mtx, dist)

    newimg = cv2.circle(frame, (imgpts[0][0][0], imgpts[0][0][1]), 5, (255, 0, 0), 5)
    newimg = cv2.circle(newimg, (imgpts[1][0][0], imgpts[1][0][1]), 5, (255, 0, 0), 5)
    newimg = cv2.circle(newimg, (imgpts[2][0][0], imgpts[2][0][1]), 5, (255, 0, 0), 5)
    newimg = cv2.circle(newimg, (imgpts[3][0][0], imgpts[3][0][1]), 5, (255, 255, 0), 5)
    imga = draw(newimg,corners2,imgpts2)

    # mean_error = 0
    # for i in range(len(objpoints)):
    #     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs, tvecs, mtx, dist)
    #     error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    #     mean_error += error

    # print("total error: ", mean_error/len(objpoints))
    k = cv2.waitKey()

    if k == ord('s'):
          cv2.imwrite(fname[:6]+'.png', imga)
    else:
          cv2.imshow('Camera output',imga)

      # k = cv2.waitKey(1) & 0xFF

print('Hey')
cv2.destroyAllWindows()