# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import cv2
from functions import *
from edge_detection import *
from matplotlib import pyplot as plt


# %%
WORKING_FOLDER = 'data/calibration/david/20-04-2021'
INTRINSIC_PATH = 'data/calibration/david/intrinsic'
RAW_TARGET_OBJECT_IMAGE_NAME = 'frame_2021-04-18 16:50:52.088963.png'
RAW_CALIBRATING_OBJECT_IMAGE_NAME = 'frame_2021-04-18 16:51:45.830721.png'
CHESSBOARD_SQUARE_LENGTH_MM = 24
CHESSBOARD_DIMENSIONS = (9,6, CHESSBOARD_SQUARE_LENGTH_MM)
PLANE_CALIBRATION_OBJECT_HEIGHT_MM = 50
UNC_COMPONENTS_MASK_SIZE = 8

# Configure plt plot sizes for notebook
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['figure.dpi'] = 200

# %% [markdown]
# ## Intrinsic parameter calibration

# %%
object_points, image_points, image_dimensions = chessboardPointExtraction(CHESSBOARD_DIMENSIONS,INTRINSIC_PATH)
ret, intrinsic_matrix, distortion_coefs, rotation_vecs, translation_vecs = cv2.calibrateCamera(object_points, image_points, image_dimensions,None,None)
print('\nImage Dimensions\n', image_dimensions)
print('\nIntrinsic Matrix\n', intrinsic_matrix)
print('\nDistortion Coefficients\n', distortion_coefs)
print(f'\nReprojection error: {calculateReprojectionError(object_points, image_points, rotation_vecs, translation_vecs, intrinsic_matrix, distortion_coefs)}')

# %% [markdown]
# ## Extrinsic parameter calibration

# %%
object_points, image_points, _ = chessboardPointExtraction(CHESSBOARD_DIMENSIONS, f'{WORKING_FOLDER}/extrinsic')
object_points_reshaped = np.array(object_points).reshape((-1,3))
image_points_reshaped = np.array(image_points).reshape((-1,2))

# Calculate extrinsic parameter matrices (translation and rotation) using PnP RANSAC
ret, rotation_vecs, translation_vecs, _ = cv2.solvePnPRansac(object_points_reshaped, image_points_reshaped, intrinsic_matrix, distortion_coefs)
print(f'Reprojection error: {calculateReprojectionError(object_points, image_points, rotation_vecs, translation_vecs, intrinsic_matrix, distortion_coefs)}')

# %% [markdown]
# ## Projection Calculation

# %%
perspective_projection_matrix = calculatePpmMatrix(intrinsic_matrix, rotation_vecs, translation_vecs)

# %% [markdown]
# ## Shadow/Light Plane Calibration

# %%

plane_calib_img_raw = cv2.imread(f'{WORKING_FOLDER}/{RAW_CALIBRATING_OBJECT_IMAGE_NAME}',cv2.IMREAD_GRAYSCALE)

# Extract shadow points from raw image (shadow points in white)
plane_calib_img_processed, _ = extract_shadow_points_auto(plane_calib_img_raw)

# Split the bottom plane shadow and the top plane shadow
processed_base_plane_points, processed_elevated_plane_points = splitTopBottomPoints(plane_calib_img_processed)

fig, subplots = plt.subplots(1,2)
subplots[0].set_title('Proc. base plane points')
subplots[0].imshow(processed_base_plane_points, cmap='gray' , vmin=0, vmax=255)

subplots[1].set_title('Proc. elevated plane points')
subplots[1].imshow(processed_elevated_plane_points, cmap='gray', vmin=0, vmax=255)


# %%
# 'Floor' points
floor_plane_constraint = [0,0,1,0] # z=0
base_points = getWhitePoint3DCoords(processed_base_plane_points, [floor_plane_constraint], perspective_projection_matrix)

# Object top points
elevated_plane_constraint = [0,0,1,PLANE_CALIBRATION_OBJECT_HEIGHT_MM]
elevated_points = getWhitePoint3DCoords(processed_elevated_plane_points, [elevated_plane_constraint], perspective_projection_matrix)

plane_3d_points = list(base_points)
plane_3d_points.extend(elevated_points)


# %%
# Calculate plane coeficients
plane_coefs = calculatePlaneCoefs(np.array(plane_3d_points))
plane_coefs

# %% [markdown]
# ## Target object point extraction

# %%
target_object_points_raw = cv2.imread(f'{WORKING_FOLDER}/{RAW_TARGET_OBJECT_IMAGE_NAME}')
target_object_points_raw_gray = cv2.cvtColor(target_object_points_raw, cv2.COLOR_BGR2GRAY)

# Detect shadow line
target_object_points_processed, _ = extract_shadow_points_auto(target_object_points_raw_gray)

# Remove unconnected components from processed image (aimed at removing some of the noise)

plt.title('Target object shadow/light points')
plt.imshow(target_object_points_processed, cmap='gray')


# %%
# Calculate y and z for points in shadow
points = getWhitePoint3DCoords(target_object_points_processed, [plane_coefs], perspective_projection_matrix)

points_x = [point[0] for point in points]
points_y = [point[1] for point in points]
points_z = [point[2] for point in points]


# %%
figure, subplots = plt.subplots(2)
subplots[0].set_title('Calculated point coordinates (view from positive x)')
subplots[0].set_xlabel('Y Coordinate')
subplots[0].set_ylabel('Z Coordinate')
subplots[0].set_xlim([0,170])
subplots[0].set_ylim([-10,60])
subplots[0].set_xticks(np.arange(0, 180, 20))
subplots[0].set_yticks(np.arange(-10, 70, 10))
subplots[0].scatter(points_y, points_z, color='black', marker='.', linewidths=0.1)

subplots[1].set_title('Calculated point coordinates (view from positive z)')
subplots[1].set_xlabel('X Coordinate')
subplots[1].set_ylabel('Y Coordinate')
subplots[1].set_xlim([-6,6])
subplots[1].set_ylim([0,170])
subplots[1].set_xticks(np.arange(-6, 7, 1))
subplots[1].set_yticks(np.arange(0, 180, 20))
subplots[1].scatter(points_x, points_y, color='black', marker='.', linewidths=0.1)
figure.tight_layout(pad=2)


# %%



