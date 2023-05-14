import cv2
import os
import sys

import math
import numpy as np
import matplotlib.pyplot as plt

import image_filters as filter
import line_filter as lf

use_camera = False
use_recording = False
use_color = 'Metal'

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
minval = 50
maxval = 90
while True:
    if use_camera or use_recording:
        ret, image = vid.read()
    else:
        path_to_img = use_color + '/' + use_color + '_' + str(index) + '.jpg'
        image = cv2.imread(path_to_img)
    ###################################################################
    cv2.imshow("original", image)
    #print(image.shape)
    #image = cv2.resize(image, [2400, 1600])
    useless, red_mask = filter.detectColorObjects(image, find_color='red')
    percent_red = np.sum(red_mask == 255) / (image.shape[1]*image.shape[1])
    #print(percent_red)

    if percent_red > red_threshold_percent:
        adjusted = cv2.convertScaleAbs(image, alpha=1, beta=0)
    else:
        adjusted = cv2.convertScaleAbs(image, alpha=10, beta=-100)

    
    cv2.imshow("adjusted", adjusted)
    #blur = cv2.bilateralFilter(adjusted, 5, 175, 175)#cv2.GaussianBlur(adjusted, (3, 3), 0)
    #blur = cv2.GaussianBlur(blur, (5, 5), 0)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    

    if percent_red > red_threshold_percent:
        precanny = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)#hsv[:,:,2]
        precanny = cv2.inRange(precanny, 0, 45)
        cv2.imshow('range', precanny)
        precanny2 = cv2.bitwise_not(filter.detectColorObjects(adjusted, find_color='red', lowsat = 50, lowval = 30)[1])
        cv2.imshow('red', precanny2)
        precanny = cv2.bitwise_and(precanny, precanny2)
    else:
        precanny = hsv[:,:,1]

    if use_camera or use_recording:
        prev_frames.append(precanny)
        if len(prev_frames) < num_frames:
            continue
        else:
            prev_frames = prev_frames[1:]

        for i in range(num_frames-1):
            precanny = cv2.bitwise_or(precanny, prev_frames[i])

    precanny = cv2.GaussianBlur(precanny, (9, 9), 0)
    if percent_red > red_threshold_percent:
        cv2.imshow("precanny", precanny)
        canny = cv2.Canny(precanny,255,255)
    else:
        cv2.imshow("precanny", precanny)
        canny = cv2.Canny(precanny,255,255)
    cv2.imshow("canny", canny) 
    """hsv_scharr = filter.edgeDetector(hsv, algo_edgedetect='scharr')


    if percent_red > red_threshold_percent:
        scharr = hsv_scharr[:,:,2]
    else:
        scharr = hsv_scharr[:,:,1]


    if use_camera or use_recording:
        prev_frames.append(scharr)
        if len(prev_frames) < num_frames:
            continue
        else:
            prev_frames = prev_frames[1:]

        for i in range(num_frames-1):
            scharr = cv2.bitwise_and(scharr, prev_frames[i])
    #cv2.imshow("rgb", rgb_scharr)
    cv2.imshow("hsv", scharr)
    blur = cv2.GaussianBlur(scharr, (5, 5), 0)
    cv2.imshow("blurred", blur) 
    canny = cv2.Canny(hsv[:,:,2],150,255)
    cv2.imshow("canny", canny) 
    #blur_scharr = cv2.GaussianBlur(gray, (7, 7), 0)
    #

    #masked = cv2.inRange(blur_scharr, 32, 255)
    #cv2.imshow("masked", masked) 

    #canny = cv2.Canny(scharr,220,255)
    #cv2.imshow("canny", canny) 

    lines, img_with_lines = filter.houghLineDetector(blur)
    #processed_lines = lf.process_lines(lines)
    #for line in processed_lines:
    #    print(line)
    #    leftx, boty = line[0]
    #    rightx, topy = line[1]
    #    cv2.line(image, (leftx, boty), (rightx,topy), (0,0,255), 6) 
    cv2.imshow("lines", img_with_lines)
    

    #img_edge = filter.edgeDetector(image, algo_edgedetect='scharr')
    #cv2.imshow("edge", img_edge) 
    
    #img_blur = cv2.GaussianBlur(image, (15, 15), 0)
    #cv2.imshow("blurred", img_blur) """

    circles = []

    contours, hierarchy = cv2.findContours(image=canny, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = filter.filterContours(contours, hierarchy)
    image_copy = image.copy()

    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow("contours", image_copy)
    for c in contours:
        try:
        
            for ellipse in filter.process_contour(c):
                if distToClosestCircle(circles, ellipse[0]) > 30 and filter.filter_ellipse(ellipse, avgsize):
                    circles.append(ellipse)
                    cv2.ellipse(image,ellipse, (0,0,255), 1)
        
        except Exception as e:
            continue
            #print(e)
            #cv2.drawContours(image=image, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow("circles", image)
    avgsize = 0.0
    for c in circles:
        avgsize += 3.1416 * c[1][0] * c[1][1]
    
    avgsize /= max(1, len(circles))


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
    #print(minval, maxval)
# After the loop release the cap object
if use_camera:
    vid.release()
# Destroy all the windows
cv2.destroyAllWindows() 