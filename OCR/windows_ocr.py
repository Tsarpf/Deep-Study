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
#%%

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string
#print(pytesseract.image_to_string(Image.open('2cQc.png')))
print('player1------------------------------------')
print(pytesseract.image_to_string(Image.open('player1.png')))

print('player2------------------------------------')
print(pytesseract.image_to_string(Image.open('player2.png')))

print('player3------------------------------------')
print(pytesseract.image_to_string(Image.open('player3.png')))

print('sitout_player1------------------------------------')
print(pytesseract.image_to_string(Image.open('sitout_player1.png')))

print('sitout_player2------------------------------------')
print(pytesseract.image_to_string(Image.open('sitout_player2.png')))

print('self------------------------------------')
print(pytesseract.image_to_string(Image.open('self.png')))

#%%

print('self------------------------------------')
print(pytesseract.image_to_string(Image.open('self.png')))


#%%
import glob
files = sorted(glob.glob('*.png'))
print(files)

#%%

for file_name in files:
    print(f'{file_name}------------------------------------')
    print(pytesseract.image_to_string(Image.open(file_name)))














#%%
# #print(pytesseract.image_to_string(Image.open('test.png')))
##  # French text image to string
##  #print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))
##  
##  # Get bounding box estimates
##  print(pytesseract.image_to_boxes(Image.open('test.png')))
##  
##  # Get verbose data including boxes, confidences, line and page numbers
##  print(pytesseract.image_to_data(Image.open('test.png')))
##  
##  # Get information about orientation and script detection
##  print(pytesseract.image_to_osd(Image.open('test.png')))
##  
##  # In order to bypass the internal image conversions, just use relative or absolute image path
##  # NOTE: If you don't use supported images, tesseract will return error
##  print(pytesseract.image_to_string('test.png'))
##  
##  # get a searchable PDF
##  pdf = pytesseract.image_to_pdf_or_hocr('test.png', extension='pdf')
##  
##  # get HOCR output
##  hocr = pytesseract.image_to_pdf_or_hocr('test.png', extension='hocr')