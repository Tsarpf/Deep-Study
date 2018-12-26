import time
import cv2
import mss
import numpy
from screeninfo import get_monitors

from segment import ore_segment

from screeninfo import get_monitors
for m in get_monitors():
    print(str(m))


monitor_size = [2560, 1440]

window_width = 860
window_height = 640

### Windows find windows
#import win32gui
from win32 import win32gui

def screenshot_and_draw(position):
    i = 0
    with mss.mss() as sct:
        img_matrix = []

        while True:
        #for _ in range(100):
            # Get raw pizels from screen and save to numpy array
            img = numpy.array(sct.grab(position))
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Save img data as matrix
            #img_matrix.append(img)
            segmented = ore_segment(img)

            # Display Image
            cv2.imshow('ses', segmented)
            print("drawn", i)
            i += 1

            # Press q to quit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        # Playback image movie from screencapture
        for img in img_matrix:
            cv2.imshow('Playback', img)
            time.sleep(0.1)

def callback(hwnd, extra):
    rect = win32gui.GetWindowRect(hwnd)
    x = int(rect[0] * 1.25)
    y = int(rect[1] * 1.25)
    w = int(rect[2] * 1.25) - x
    h = int(rect[3] * 1.25) - y
    text = win32gui.GetWindowText(hwnd)
    if text == "Old School RuneScape":
        print("Window %s:" % text)
        print("\tLocation: (%d, %d)" % (x, y))
        print("\t    Size: (%d, %d)" % (w, h))

        size = [
            int(w),
            int(h)]
        position = [x, y]
        monitor = {
            'top': y,
            'left': x,
            'width': size[0],
            'height': size[1]
        }
        screenshot_and_draw(monitor)

win32gui.EnumWindows(callback, None)

