# 3D Acquisition - VCOM

**2020/2021** - 4th Year, 2nd Semester

**Course:** *Visão por Computador* ([VCOM](https://sigarra.up.pt/feup/en/ucurr_geral.ficha_uc_view?pv_ocorrencia_id=384980)) | Computer Vision
**Authors:** Bernardo Santos([bernas670](https://github.com/bernas670)), David Silva ([daviddias99](https://github.com/daviddias99)), Laura Majer ([m-ajer](https://github.com/m-ajer)) Luís Cunha ([luispcunha](https://github.com/luispcunha))

---

**Description:** In this work we use a structured light technique to implement a system for 3D data acquisition, namely information about the height of objects, using household items. The technique consists of casting a pattern of light/shadow over the objects, capturing an image of said objects and, after detecting and extracting the pattern of the line, using the perspective projection matrix to compute the real world coordinates of the pattern points. With the image coordinate of those points and the projection matrix we are only able to obtain the line of sight of the image point, but by constraining the plane (light or shadow) to which the points of the pattern belong we are able to obtain the real world coordinates of each of the points of the pattern. We explore several computer vision techniques, including edge detection, data acquisition, and camera calibration. We are able to measure objects that do not have significant texture with a satisfactory precision, as well as introduce some degree of automation to the selection of suitable parameters for the algorithms used.

For more information on the specification see `docs/specification.pdf` and for a detailed report on our work see `docs/report.pdf`.

**Technologies:** Python, OpenCV, Jupyter Notebooks

**Skills:** Camera calibration, 3D data acquisition, linear algebra, edge detection, classical computer vision, line detection, structured light, acquisition setup

**Grade:** 18.5/20

---

## Help

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