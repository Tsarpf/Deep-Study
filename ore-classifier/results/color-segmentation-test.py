#0C0503
#381D14
# ^ not really used
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import os
cwd = os.getcwd()
print(cwd)


flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

varrock_small = cv2.imread("./ore-classifier/varrock_south_west.png")
varrock_small = cv2.cvtColor(varrock_small, cv2.COLOR_BGR2RGB)
#plt.imshow(varrock_small)
#plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

pixel_colors = varrock_small.reshape(
    (np.shape(varrock_small)[0]*np.shape(varrock_small)[1],3))

norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


def draw_color_scatter(target_image):
    r, g, b = cv2.split(target_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()
    print('drawn')

#draw_color_scatter(varrock_small)


hsv_varrock_small = cv2.cvtColor(varrock_small, cv2.COLOR_RGB2HSV)
def draw_hsv_scatter(target_image):
    h, s, v = cv2.split(target_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    #axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()


#draw_hsv_scatter(hsv_varrock_small)


def show_hsv_colors(light, dark):
    from matplotlib.colors import hsv_to_rgb
    lo_square = np.full((10, 10, 3), light, dtype=np.uint8) / 255.0
    do_square = np.full((10, 10, 3), dark, dtype=np.uint8) / 255.0
    plt.subplot(1, 2, 1)
    plt.imshow(hsv_to_rgb(lo_square))
    plt.subplot(1, 2, 2)
    plt.imshow(hsv_to_rgb(do_square))
    plt.show()


hue_max = 179
sat_max = 255
val_max = 255

iron_l = (12, 0.64, 0.26)
iron_d = (21, 0.76, 0.04)

# coal
# light 53.33° 40% 17.65%
# dark 50° 85.71% 5.49%
#coal_l = (53.33, 0.4, 0.1765)
#coal_d = (50, 0.8571, 0.0549)j

# light2 60, 41%, 18%
# dark2 60, 36%, 5%
#coal_l = (55.7, 0.36, 0.18)
#coal_d = (60, 0.48, 0.05)

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
#print(lower, upper)
#show_hsv_colors(light_iron, dark_iron)

def compare(result, mask):
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()

# define some kernels for morph operations
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
elliptical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
one_kernel = np.ones((2, 2), np.uint8)

# with colored mask
#mask = cv2.inRange(hsv_varrock_small, lower_coal, upper_coal)
mask = cv2.inRange(hsv_varrock_small, lower_iron, upper_iron)
result = cv2.bitwise_and(varrock_small, varrock_small, mask=mask)
compare(result, mask)

# erode individual pixels
erosion_mask = cv2.erode(mask, one_kernel, iterations = 2)
result_eroded = cv2.bitwise_and(varrock_small, varrock_small, mask=erosion_mask)
compare(result_eroded, erosion_mask)

# with morph closed mask
closed_mask = cv2.morphologyEx(erosion_mask, cv2.MORPH_CLOSE, rect_kernel)
result_closed = cv2.bitwise_and(varrock_small, varrock_small, mask=closed_mask)
compare(result_closed, closed_mask)

# erode individual pixels
kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(closed_mask, kernel, iterations = 1)
result_eroded = cv2.bitwise_and(varrock_small, varrock_small, mask=erosion)
compare(result_eroded, erosion)

# with dilated and closed mask
#dilation_mask = cv2.dilate(closed_mask, rect_kernel, iterations = 1)
#result_dilated = cv2.bitwise_and(varrock_small, varrock_small, mask=dilation_mask)
#result_with_mask(result_dilated, dilation_mask)

compare(erosion, varrock_small)



print('done')
