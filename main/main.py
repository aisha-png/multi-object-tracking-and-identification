import cv2 as cv
import numpy as np

# load images
haystack_img = cv.imread('assets/ac_pumpkins.png', cv. IMREAD_UNCHANGED)
needle_img = cv.imread('assets/ac_single_pumpkin.png', cv.IMREAD_UNCHANGED)

# call match template (cv.TM_CCOEFF_NORMED seems to give the better result) // comparison method
result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

# cv.imshow('Result', result)
# cv.waitKey()
