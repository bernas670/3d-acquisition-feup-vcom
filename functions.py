import cv2
import numpy as np
import glob
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor, LinearRegression

# Find the chessboard corners subpixel coordinates of a given set of images
def chessboardPointExtraction(chessboard_dimensions, frame_path):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (0,1,0), (0,2,0) ....,(5,6,0). The coordinates are adjusted according to the chessboard square side length
    sample_object_points = np.zeros(
        (chessboard_dimensions[0] * chessboard_dimensions[1], 3), np.float32)

    sample_object_points[:, :2] = np.mgrid[0:chessboard_dimensions[0],
                                           0:chessboard_dimensions[1]].T.reshape(-1, 2) * chessboard_dimensions[2]

    # Swap axis so that the z axis is perpendicular to the chessboard
    sample_object_points[:, [1, 0]] = sample_object_points[:, [0, 1]]

    # Arrays to store object points and image points from all the images.
    object_points = []  # 3d point in real world space
    image_points = []  # 2d points in image plane.

    images = glob.glob(f'{frame_path}/*.png')
    print(f'Found {len(images)} calibration images')

    for fname in tqdm(images):
        image = cv2.imread(fname)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            grayscale_image, (chessboard_dimensions[0], chessboard_dimensions[1]), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            object_points.append(sample_object_points)

            # Subpixel location
            improved_corners = cv2.cornerSubPix(
                grayscale_image, corners, (11, 11), (-1, -1), criteria)
            image_points.append(improved_corners)
        else:
            print(f'Could not find chessboard corners in image {fname}')
    print('\nCalibration over')
    return object_points, image_points, grayscale_image.shape[::-1]


def calculateReprojectionError(objpoints, imgpoints, rvecs, tvecs, imatrix, distortion):
    mean_error = 0
    for i in range(len(objpoints)):

        # Calibrate camera returns several rotation and translation vectors
        # but solvePnP only returns one

        rot_vecs = rvecs if len(rvecs) == 3 else rvecs[i]
        tra_vecs = tvecs if len(tvecs) == 3 else tvecs[i]

        calculted_image_points, _ = cv2.projectPoints(objpoints[i], rot_vecs, tra_vecs, imatrix, distortion)

        error = cv2.norm(imgpoints[i], calculted_image_points,
                         cv2.NORM_L2)/len(calculted_image_points)
        mean_error += error

    return mean_error/len(objpoints)

# Get xyz coordinates of an i,j image point given a perpective projection matrix and a set of aditional constraints
# Constraints is a list where each element is a list [A,B,C,D] representing a constraint of type Ax + By + Cz = D
def get_xyz_coords(i, j, ppm, constraints=[[1,0,0,0]]):
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
    
    # rotation_vecs is in Rodrigues forms, and needs to be converted to a 3x3 matrix
    rotation_matrix = cv2.Rodrigues(rotation_vecs)[0]

    # create [R|T] matrix
    extrinsic_matrix = np.concatenate((rotation_matrix, translation_vecs), axis=1)

    # multiply the intrinsic and extrinsinc matrix
    perspective_projection_matrix = intrinsic_matrix @ extrinsic_matrix

    return perspective_projection_matrix


# The points that are used to calculate the shadow plane are in different z planes. This function
# only allows the regression to be done with points that are not on the same z plane. Ideally, the 
# meausred points with the same z coordinate should be colinear and, thus, shouldn't be considered, 
# but to to imperfections in the 3D coordinate this is not the case.
def validateData(_, b):
    return not(b[0] == b[1] and b[1] == b[2])

# Calculate the coeffiecents of a plane that fits the given <points>. The estimation is done using
# RANSAC. The algorithm expects points in to different z planes. The coeficients are returned in the
# form [A,B,C,D] where the plane is of the form Ax+BY+CZ = -D
def calculatePlaneCoefs(points):
    xy = points[:, :2]
    z = points[:, 2]

    # estimate Ax + By + D = Z (C = 1)
    # validateData ensures that points used to estimate the plane are in different Z planes
    # without this the algorithm was considering all points in other planes as outliers
    reg = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=True),
                          is_data_valid=validateData,
                          residual_threshold=0.01,
                          max_trials=1000).fit(xy, z)

    return list(np.append(reg.estimator_.coef_, [-1, -reg.estimator_.intercept_]))


# Calculate the 3D coordinates of the write 2D <points>, given a set of extra constraints (see get_xyz_coords) and a
# perspective projection matrix
def getWhitePoint3DCoords(points, constraints, ppm):
    white_pixel_coords = cv2.findNonZero(points)
    res = [get_xyz_coords(pixel[0, 0], pixel[0, 1], ppm, constraints) for pixel in white_pixel_coords]

    return np.array(res)

# Calculate straight line that fits <points>. Returns the slope, y intercept and inlier mask
def calculateLineCoefs(points):
    x = points[:, :1]
    y = points[:, 1]

    # estimate Ax + B = y 
    reg = RANSACRegressor(base_estimator=LinearRegression(fit_intercept=True)).fit(x, y)

    return reg.estimator_.coef_, reg.estimator_.intercept_, reg.inlier_mask_

# TODO: como isto n é deterministico, as vezes n dá mt bem
# - ideia: fazer um validate_model
def splitTopBottomPoints(src):
    points = cv2.findNonZero(src)

    _, _, inliers = calculateLineCoefs(np.reshape(points, (-1, 2)))

    bottom = np.zeros_like(src, dtype=np.uint8)
    for pt in points[inliers]:
        bottom[pt[0, 1], pt[0, 0]] = 255

    points_top = points[np.logical_not(inliers)]

    _, _, inliers = calculateLineCoefs(np.reshape(points_top, (-1, 2)))
    top = np.zeros_like(src, dtype=np.uint8)
    for pt in points_top[inliers]:
        top[pt[0, 1], pt[0, 0]] = 255

    return bottom, top
