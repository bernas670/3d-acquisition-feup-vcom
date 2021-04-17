import cv2
import numpy as np
import glob
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor, LinearRegression


def chessboardPointExtraction(chessboard_dimensions, frame_path):
  # termination criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  sample_object_points = np.zeros((chessboard_dimensions[0] * chessboard_dimensions[1],3), np.float32)

  # TODO: check if this is the correct way to adjust the sizes. Used this accordint to: https://stackoverflow.com/questions/37310210/camera-calibration-with-opencv-how-to-adjust-chessboard-square-size
  sample_object_points[:,:2] = np.mgrid[0:chessboard_dimensions[0],0:chessboard_dimensions[1]].T.reshape(-1,2) * chessboard_dimensions[2]

  # Swap axis so that the z axis is perpendicular to the chessboard
  sample_object_points[:, [1, 0]] = sample_object_points[:, [0, 1]]

  # Arrays to store object points and image points from all the images.
  object_points = [] # 3d point in real world space
  image_points = [] # 2d points in image plane.

  images = glob.glob(f'{frame_path}/*.png')
  print(f'Found {len(images)} calibration images')

  for fname in tqdm(images):
      image = cv2.imread(fname)
      grayscale_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

      # Find the chess board corners
      ret, corners = cv2.findChessboardCorners(grayscale_image, (chessboard_dimensions[0], chessboard_dimensions[1]), None)

      # If found, add object points, image points (after refining them)
      if ret == True:
          object_points.append(sample_object_points)

          improved_corners = cv2.cornerSubPix(grayscale_image, corners, (11,11), (-1,-1), criteria)
          image_points.append(improved_corners)
      else:
          print(f'Could not find chessboard corners in image {fname}')
  print('\nCalibration over')
  return object_points, image_points, grayscale_image.shape[::-1]

def calculateReprojectionError(objpoints, imgpoints, rvecs, tvecs, imatrix, distortion):
  mean_error = 0
  for i in range(len(objpoints)):

      if(len(rvecs) == 3):
        calculted_image_points, _ = cv2.projectPoints(objpoints[i], rvecs, tvecs, imatrix, distortion)
      else:
        calculted_image_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], imatrix, distortion)
      error = cv2.norm(imgpoints[i], calculted_image_points, cv2.NORM_L2)/len(calculted_image_points)
      mean_error += error

  return mean_error/len(objpoints)


# Constraints is a list where each element is a list [A,B,C,D] representing a constraint of type Ax + By + Cz = D
def get_xyz_coords(i, j, ppm, constraints = None):
  k1 = ppm[2] * i    
  k2 = ppm[2] * j
  k3 = k1 - ppm[0]
  k4 = k2 - ppm[1]
  a = [k3[0:-1], k4[0:-1]]
  b = [-k3[-1], -k4[-1]]

  for constraint in constraints:
      a.append(np.array(constraint[0:-1]))
      b.append(np.array(constraint[-1]))
  res = np.linalg.solve(a, b)

  return res

def calculatePpmMatrix(intrinsic_matrix, rotation_vecs, translation_vecs):
  rotation_matrix = cv2.Rodrigues(rotation_vecs)[0]
  extrinsic_matrix = np.concatenate((rotation_matrix, translation_vecs), axis=1)
  perspective_projection_matrix = intrinsic_matrix @ extrinsic_matrix

  return perspective_projection_matrix

def calculatePlaneCoefs(points):
  xy = points[:, :2]
  z = points[:, 2]

  # estimate Ax + By + D = Z (C = 1)
  reg = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=True), residual_threshold=0.01).fit(xy, z)

  return list(np.append(reg.estimator_.coef_,[-1, -reg.estimator_.intercept_])) 

def getWhitePoint3DCoords(points, constraints, ppm):
  res = []
  white_pixel_coords = cv2.findNonZero(points)
  for pixel in white_pixel_coords:
    point = get_xyz_coords(pixel[0,0],pixel[0,1], ppm, constraints)
    res.append(point)

  return res