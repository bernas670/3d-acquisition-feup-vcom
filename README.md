# 3D Acquisition - VCOM
Code developed for the Computer Vision course @FEUP

To run the experiments please use the `experiments.ipynb` notebook (alternatively you can run the `script.py` file, however, for viewing the plots we advise using the notebook version).

The second cell contains variables that need to be configured to run the experiments:

```python
WORKING_FOLDER = 'data/calibration/david/20-04-2021'
INTRINSIC_PATH = 'data/calibration/david/intrinsic'
RAW_TARGET_OBJECT_IMAGE_NAME = 'frame_2021-04-18 16:51:45.830721.png'
RAW_CALIBRATING_OBJECT_IMAGE_NAME = 'frame_2021-04-18 16:51:45.830721.png'
CHESSBOARD_SQUARE_LENGTH_MM = 24
CHESSBOARD_DIMENSIONS = (9,6, CHESSBOARD_SQUARE_LENGTH_MM)
PLANE_CALIBRATION_OBJECT_HEIGHT_MM = 50
UNC_COMPONENTS_MASK_SIZE = 8
```

Used libraries:

* numpy
* opencv
* sklearn
* matplotlib
* tqdm

The `edge_detection.py` and `functions.py` contain the functions used during the methodology. These functions have been commented to aid perception.

The `data` folder contains an example of images that can be used in the methodology. To run it, the user should provide a folder (`WORKING_FOLDER`) with the following structure.

```
<WORKING_DIRECTORY>/
├─ extrinsic/
│  ├─ ex1.png
│  ├─ .../
├─ <RAW_TARGET_OBJECT_IMAGE_NAME>
├─ <RAW_CALIBRATING_OBJECT_IMAGE_NAME>
```

The `helper` folder contains several scripts that were used during the development process. We can't garantee that these files will function correctly from the get-go.