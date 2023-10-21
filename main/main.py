import cv2
import numpy as np

# print("OpenCV version:", cv2.__version__)
# load images
haystack_img = cv2.imread('assets/albion_farm.jpeg', cv2. IMREAD_UNCHANGED)
needle_img = cv2.imread('assets/albion_cabbage.jpeg', cv2.IMREAD_UNCHANGED)

# call match template (cv.TM_CCOEFF_NORMED seems to give the better result) // comparison method
result = cv2.matchTemplate(haystack_img, needle_img, cv2.TM_CCOEFF_NORMED)

# cv2.imshow('Result', result)
# cv2.waitKey()

# function: min max location
# get best match position
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print('Best match top left position: %s' % str(max_loc))
print('Best match confidence: %s' % max_val)
# Best match top left position: (685, 398)
# Best match confidence: 0.7784304618835449 - not a good percentage which it should be higher

# to verify we found image:
threshold = 0.8
if max_val >= threshold:
    print('Needle found.')

    # get dimensions of the needle image
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)
    # (x coordinate of top_left and + width of img, and y coordinate of top_left and + height of img)

    cv2.rectangle(haystack_img, top_left, bottom_right, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)
#     color = (red, green, blue) - so the lines should appear green

else:
    print('Needle not found.')
