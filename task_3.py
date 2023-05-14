import cv2
import os
import sys

import math
import numpy as np
import matplotlib.pyplot as plt

import image_filters as filter
import line_filter as lf

use_camera = False
use_recording = True
use_color = 'Red'

red_threshold_percent = 0.03
if use_color == 'White':
    index_max = 14
    num_frames = 40
elif use_color == 'Red':
    index_max = 4
    num_frames = 4
else:
    index_max = 6

if use_camera:
    vid = cv2.VideoCapture(0)
if use_recording:
    if use_color == "Red":
        vid = cv2.VideoCapture('Red/video_red_no_holes_covered.mov')
    else:
        vid = cv2.VideoCapture('White/video_white_no_holes_covered.mov')
index = 1

prev_frames = []

def distToClosestCircle(circles, point):
  dist = 10000
  for c in circles:
    new_dist = math.sqrt((point[0] - c[0][0])**2 + (point[1] - c[0][1])**2)
    if new_dist < dist:
      dist = new_dist

  return dist

avgsize = 1
minval = 0
maxval = 50
contourIndex = 0
while True:
    if use_camera or use_recording:
        ret, image = vid.read()
    else:
        path_to_img = use_color + '/' + use_color + '_' + str(index) + '.jpg'
        image = cv2.imread(path_to_img)
    ###################################################################
    
    cv2.imshow("original", image)
    #print(image.shape)
    #image = cv2.resize(image, [720, 480])
    blur = cv2.bilateralFilter(image, 5, 255, 255)
    #cv2.imshow('blur', blur)
    canny = cv2.Canny(blur,20,50)
    lines = [1, 1]
    cv2.imshow("canny", canny)
    lines, img_with_lines = filter.houghLineDetector(canny)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(canny,(x1,y1),(x2,y2),(0,255,0),5)
    #cv2.imshow("canny no lines", canny)
    contours, hierarchy = cv2.findContours(image=canny, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = filter.filterContours(contours, hierarchy)
    image_copy = image.copy()

    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    
    circles = []
    for c in contours:
        ellipses, image_copy2 = filter.process_contour(c, image_copy)
    cv2.imshow("contours", image_copy)

    #################################################################
    keypress = cv2.waitKey(1)
    if keypress & 0xFF == ord('q'):
        break
    elif keypress & 0xFF == ord('a'):
        index -= 1
        if index == 0:
            index = index_max
    elif keypress & 0xFF == ord('d'):
        index += 1
        if index > index_max:
            index = 1
    elif keypress & 0xFF == ord('r'):
        minval += 5
    elif keypress & 0xFF == ord('f'):
        minval -= 5
    elif keypress & 0xFF == ord('t'):
        maxval += 5
    elif keypress & 0xFF == ord('g'):
        maxval -= 5
    elif keypress & 0xFF == ord('1'):
        contourIndex -= 1
        if contourIndex < 0:
            contourIndex = len(contours) - 1
    elif keypress & 0xFF == ord('2'):
        contourIndex += 1
        if contourIndex >= len(contours):
            contourIndex = 0
    #print(contourIndex, len(contours))
    #print(minval, maxval)
# After the loop release the cap object
if use_camera:
    vid.release()
# Destroy all the windows
cv2.destroyAllWindows() 