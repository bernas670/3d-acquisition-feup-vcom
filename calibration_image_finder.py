from itertools import count
from os import path
from datetime import datetime
import numpy as np
import cv2

# Calibration parameters
CAPTURE_DEVICE = 0
FRAME_COUNT = 20
WAIT_PERIOD_MS = 5000
CAMERA_ID = 'david'
FRAME_FULL_PATH = path.join(path.abspath(''), 'data', 'calibration', CAMERA_ID)
captureDevice = cv2.VideoCapture(CAPTURE_DEVICE)

ret, frame = captureDevice.read()
cv2.imshow('Camera output',frame)
for currentFrameNumber in range(FRAME_COUNT):
    print(f'Waiting for frame {currentFrameNumber}')
    ret, frame = captureDevice.read()
    cv2.imshow('Camera output',frame)
    pressedKey = cv2.waitKey(WAIT_PERIOD_MS)

    if pressedKey != -1 and chr(pressedKey) == 'q':
        break

    cv2.imwrite(path.join(FRAME_FULL_PATH, f'calibration_frame_{datetime.now()}.png'), frame)

cv2.destroyAllWindows()
