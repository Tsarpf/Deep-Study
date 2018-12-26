import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# opencv hsv ranges
hue_max = 179
sat_max = 255
val_max = 255

# hand-picked from varrock_south_west.png
iron_l = (12, 0.64, 0.26)
iron_d = (21, 0.76, 0.04)

# hand-picked from barbarian_village.png
coal_l = (55, 0.35, 0.19)
coal_d = (61, 0.49, 0.03)

# convert to 255
light_iron = (
    int(iron_l[0] / 360 * hue_max),
    int(iron_l[1] * sat_max),
    int(iron_l[2] * val_max)
)
dark_iron = (
    int(iron_d[0] / 360 * hue_max),
    int(iron_d[1] * sat_max),
    int(iron_d[2] * val_max)
)

light_coal = (
    int(coal_l[0] / 360 * hue_max),
    int(coal_l[1] * sat_max),
    int(coal_l[2] * val_max)
)
dark_coal = (
    int(coal_d[0] / 360 * hue_max),
    int(coal_d[1] * sat_max),
    int(coal_d[2] * val_max)
)

# get lower/upper bounds for inRange
lower_iron = (
    min(light_iron[0], dark_iron[0]),
    min(light_iron[1], dark_iron[1]),
    min(light_iron[2], dark_iron[2])
)
upper_iron = (
    max(light_iron[0], dark_iron[0]),
    max(light_iron[1], dark_iron[1]),
    max(light_iron[2], dark_iron[2])
)

lower_coal = (
    min(light_coal[0], dark_coal[0]),
    min(light_coal[1], dark_coal[1]),
    min(light_coal[2], dark_coal[2])
)
upper_coal = (
    max(light_coal[0], dark_coal[0]),
    max(light_coal[1], dark_coal[1]),
    max(light_coal[2], dark_coal[2])
)


# define some kernels for morh operations
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
#elliptical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

ores = {
    'iron': [lower_iron, upper_iron],
    'coal': [lower_coal, upper_coal],
}

def ore_segment(image, ore):
    lower = ores[ore][0]
    upper = ores[ore][1]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # color mask
    mask = cv2.inRange(hsv_image, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    # erode 1 pixels first
    one_kernel = np.ones((2, 2), np.uint8)
    erosion_mask = cv2.erode(mask, one_kernel, iterations = 2)
    result_eroded = cv2.bitwise_and(image, image, mask=erosion_mask)

    # morph closed mask
    closed_mask = cv2.morphologyEx(erosion_mask, cv2.MORPH_CLOSE, rect_kernel)
    result_closed = cv2.bitwise_and(image, image, mask=closed_mask)

    # erode individual pixels possibly left
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(closed_mask, kernel, iterations = 1)
    result_eroded = cv2.bitwise_and(image, image, mask=erosion)

    return erosion



print('done')
