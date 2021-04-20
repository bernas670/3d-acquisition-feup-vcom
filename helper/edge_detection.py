import os
import math
import argparse
import cv2 as cv
import numpy as np


def extract_shadow_points(img, high_thresh, low_thresh, dilate, erode, after_canny=None, after_dilate=None):
    # Size of the Sobel kernel used to compute the derivatives in the Canny edge detector
    aperture = 3

    # Apply Canny edge detector
    img_canny = after_canny
    if after_dilate is None and after_canny is None:
        img_canny = cv.Canny(img, low_thresh, high_thresh,
                             apertureSize=aperture)

    # Dilate with a square kernel of size "dilate", in attempt to join both edges of the shadow
    img_dilated = after_dilate
    if after_dilate is None:
        img_dilated = cv.morphologyEx(
            img_canny, cv.MORPH_DILATE, np.ones((dilate, dilate)))

        # Erode with a square kernel of size "erode", in attempt to remove every white pixel, except for the ones in the center of the shadow
    img_morph = cv.morphologyEx(img_dilated, cv.MORPH_ERODE,
                                np.ones((erode, erode)))

    # Apply a dilation operation to the center of the shadow obtained in the previous step. The kernel
    # used is a square kernel, larger than the used in the previous dilate operation by 1, with only
    # the elements of the bottom half of the kernel set to 1, and the rest to 0.
    #
    # Example:
    # 0 0 0        0 0 0 0
    # 0 0 0   or   0 0 0 0
    # 1 1 1        1 1 1 1
    #              1 1 1 1
    #
    # Applying this dilation results in the "growth" of the detected line upwards.
    kernel_size = dilate + 1

    # Obtain kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    kernel[int(kernel_size/2):, :] = 1

    # Compute dilation with the obtained kernel
    img_dilate_up = cv.morphologyEx(img_morph, cv.MORPH_DILATE, kernel)

    # Compute the final result by considering only the pixels that are both white in the original output
    # of the canny edge detector and white in the image obtained in the previous step. This selects pixels
    # that are from the "top" edge of the shadow,
    # which are the extracted shadow points used in later stages.
    final_res = cv.bitwise_and(img_canny, img_dilate_up)

    return final_res, img_dilate_up, img_morph, img_dilated, img_canny


def remove_unconnected_points(img, area_threshold):
    # Find connected components from the image and the number of pixels in each of the components

    connectivity = 8  # find 8-way connected components

    num_labels, labels_im, stats, _ = cv.connectedComponentsWithStats(
        img, connectivity, stats=cv.CC_STAT_AREA)

    # remove the components with less than "area_threshold" pixels
    for i in range(num_labels):
        if stats[i, cv.CC_STAT_AREA] < area_threshold:
            img[labels_im == i] = 0

    return img


def evaluate(img):
    """
    Compute score of the obtained shadow points.
    The score consists of the the summation of, for each column of the image, the absolute value of the
    number of white points minus 1
    """
    score = 0
    for i in range(img.shape[1]):
        score += abs(np.count_nonzero(img[:, i]) - 1)
    return score


def extract_shadow_points_auto(img):
    """
    Extract shadow points from image, by automatically finding set of hyperparameters that obtain a good
    result according to the evaluate function above.
    """
    best = {
        'params': {
            'low_threshold': 0,
            'high_threshold': 0,
            'dilate': 0,
            'erode': 0,
        },
        'steps': {
            'after_canny': np.zeros_like(img),
            'after_dilate': np.zeros_like(img),
            'after_morph': np.zeros_like(img),
            'after_dilate_up': np.zeros_like(img),
        },
        'result': np.zeros_like(img),
        'score': math.inf
    }

    # high threshold values
    for ht in range(100, 150, 5):
        print(ht)
        # low threshold values
        for lt in range(int(0.6*ht), int(0.9*ht), 3):
            after_canny = None
            # dilation kernel size values
            for d in range(9, 15, 2):
                after_dilate = None
                # erosion kernel size values
                for e in range(d+4, d+8, 2):

                    # apply shadow points extraction algorithm
                    res, after_dilate_up, after_morph, after_dilate, after_canny = extract_shadow_points(
                        img, ht, lt, d, e, after_canny=after_canny, after_dilate=after_dilate)

                    # evaluate shadow points and see if it's the best so far
                    curr_value = evaluate(res)

                    if curr_value < best['score']:
                        best['params']['dilate'] = d
                        best['params']['erode'] = e
                        best['params']['high_threshold'] = ht
                        best['params']['low_threshold'] = lt
                        best['score'] = curr_value
                        best['result'] = res
                        best['steps']['after_canny'] = after_canny
                        best['steps']['after_morph'] = after_morph
                        best['steps']['after_dilate'] = after_dilate
                        best['steps']['after_dilate_up'] = after_dilate_up

    return best


######
##
# Auxiliary functions for manually tweaking the hyperparameters, using trackbars
##
######
edge_detection_window = 'Step 1. Canny edge detector'
morph_window = 'Step 2. Dilate -> Erode'
dilate_up_window = 'Step 3. Dilate up'
final_window = 'Final Result'


def display(final_result, after_dilate_up, after_morph_ops, after_dilate, after_edge_detection):
    cv.imshow(edge_detection_window, after_edge_detection)
    cv.imshow(morph_window, after_morph_ops)
    cv.imshow(dilate_up_window, after_dilate_up)
    cv.imshow(final_window, final_result)


# Initial values for the trackbars
low_threshold = 100
high_threshold = 150
dilate = 10
erode = 18


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract shadow points.')
    parser.add_argument('--path', dest='img_path', type=str, help='Path to image.')
    parser.add_argument('--auto', action='store_true',
                        help='whether to find parameters automatically or use a trackbar to define them manually')

    args = parser.parse_args()
    print(args)
    if args.img_path is None:
        img_path = '/home/luispcunha/repos/feup/vcom/vcom-proj1/data/cube.png'
    else:
        img_path = args.img_path

    name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv.imread(
        img_path, cv.IMREAD_GRAYSCALE)

    if args.auto:
        best = extract_shadow_points_auto(img)

        cv.imshow(final_window, best['result'])
        print(best['params'])

        cv.waitKey(0)
        exit()

    # Max values for the trackbars
    morph_max = 30
    threshold_max = 300

    cv.namedWindow(edge_detection_window)
    cv.namedWindow(morph_window)
    cv.namedWindow(dilate_up_window)
    cv.namedWindow(final_window)

    # Trackbars for controlling hyperparameters
    cv.createTrackbar('Low Threshold', edge_detection_window,
                      low_threshold, threshold_max, on_low_threshold)
    cv.createTrackbar('High Threshold', edge_detection_window,
                      high_threshold, threshold_max, on_high_threshold)
    cv.createTrackbar('Dilate', morph_window,
                      dilate, morph_max, on_dilate)
    cv.createTrackbar('Erode', morph_window,
                      erode, morph_max, on_erode)

    while True:
        results = extract_shadow_points(
            img, high_threshold, low_threshold, dilate, erode)
        display(*results)

        key = cv.waitKey(100)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv.imwrite('{}_result_{}_{}_{}_{}.png'.format(name,
                                                          low_threshold, high_threshold, dilate, erode), results[0])

    cv.destroyAllWindows()
