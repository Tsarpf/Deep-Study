#%%
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

import os
print('current path', os.getcwd())
path = r'C:\devaus\Deep-Study\OCR'
os.chdir(path)
print('current path', os.getcwd())
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
#%%
print('player1------------------------------------')
print(pytesseract.image_to_string(Image.open('player1.png')))

#%%
import glob
files = sorted(glob.glob('*.png'))
print(files)

#%%

for file_name in files:
    print(f'{file_name}------------------------------------')
    print(pytesseract.image_to_string(Image.open(file_name)))



#%%
import cv2
import numpy as np

filename = 'bb_player_stack'
image = cv2.imread(f'{filename}.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
gray = cv2.medianBlur(gray, 3)

# erode 1 pixels first
one_kernel = np.ones((3, 3), np.uint8)
erosion_mask = cv2.erode(gray, one_kernel, iterations = 1)
result_eroded = cv2.bitwise_and(gray, gray, mask=erosion_mask)

filename_gray = "{}.png".format(f'{filename}-gray')
cv2.imwrite(filename_gray, gray)

filename_eroded = "{}.png".format(f'{filename}-gray-eroded')
cv2.imwrite(filename_eroded, result_eroded)

#%%
print('gray------------------------------------')
print(pytesseract.image_to_string(Image.open('bb_player_stack-gray.png'), lang=None, config="psm 7"))
print('gray eroded------------------------------------')
print(pytesseract.image_to_string(Image.open('bb_player_stack-gray-eroded.png'), lang=None, config="psm 7"))


#%%
