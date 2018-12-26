import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# opencv vals
hue_max = 179
sat_max = 255
val_max = 255

# hand-picked from varrock_south_west.PNG
li = (18, 0.617, 0.1843)
di = (21, 0.8235, 0.0667)

# convert to 255
light_iron = (
    int(li[0] / 360 * hue_max),
    int(li[1] * sat_max),
    int(li[2] * val_max)
)
dark_iron = (
    int(di[0] / 360 * hue_max),
    int(di[1] * sat_max),
    int(di[2] * val_max)
)

# get lower/upper bounds for inRange
lower = (
    min(light_iron[0], dark_iron[0]),
    min(light_iron[1], dark_iron[1]),
    min(light_iron[2], dark_iron[2])
)
upper = (
    max(light_iron[0], dark_iron[0]),
    max(light_iron[1], dark_iron[1]),
    max(light_iron[2], dark_iron[2])
)

# define some kernels for morh operations
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
#elliptical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))


def ore_segment(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # color mask
    mask = cv2.inRange(hsv_image, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    # morph closed mask
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, rect_kernel)
    result_closed = cv2.bitwise_and(image, image, mask=closed_mask)

    # erode individual pixels possibly left
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(closed_mask, kernel, iterations = 1)
    result_eroded = cv2.bitwise_and(image, image, mask=erosion)

    return erosion



print('done')
