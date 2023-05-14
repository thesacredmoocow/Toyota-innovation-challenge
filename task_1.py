# import dependencies
import os
import sys

import math
import numpy as np
import matplotlib.pyplot as plt

import cv2

from base64 import b64decode, b64encode
import PIL
import io
import html
import time

path_to_img = '/content/gdrive/MyDrive/Toyota Innovation Challenge Documents/Training Images/Metal/Metal_2.jpg'
algo_edgedetect = ['sobel', 'prewitt','canny','roberts','scharr']
def edgeDetector(img,algo_edgedetect=None):
    if algo_edgedetect == 'canny':
        img_edge = cv2.Canny(img,100,200)
    if algo_edgedetect == 'sobel':
        img_edge = cv2.Sobel(img,-1,1,0,ksize=-1)
    if algo_edgedetect == 'prewitt':
        prewitt_kernel_h = np.array([[1,1,1],
                                    [0,0,0],
                                    [-1,-1,-1]])
        prewitt_kernel_v = np.array([[-1,0,1],
                                    [-1,0,1],
                                    [-1,0,1]])
        prewitt_h = cv2.filter2D(img,-1,prewitt_kernel_h)
        prewitt_v = cv2.filter2D(img,-1,prewitt_kernel_v)
        img_edge = prewitt_h + prewitt_v
    if algo_edgedetect == 'roberts':
        roberts_kernel_h = np.array([[-1,0],[0,1]])
        roberts_kernel_v = np.array([[0,-1],[1,0]])
        roberts_h_ele = cv2.filter2D(img,-1,roberts_kernel_h)
        roberts_v_ele = cv2.filter2D(img,-1,roberts_kernel_v)
        img_edge = roberts_h_ele + roberts_v_ele
    if algo_edgedetect == 'scharr':
        # img_gaussian = cv2.GaussianBlur(img,(3,3),5)
        # img_edge = cv2.Laplacian(img_gaussian,-1)
        img_edge = cv2.Scharr(img,-1,1,0)
    # print(img_edge.dtype0
    plt.figure(figsize=(20,10))
    plt.subplot(121),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.title('Input',color='c')
    plt.subplot(122),plt.imshow(cv2.cvtColor(img_edge,cv2.COLOR_BGR2RGB)),plt.title('Output',color='c')
    #plt.show()
    return img_edge

def detectBlobs(img):
  params = cv2.SimpleBlobDetector_Params()

  # Filter by Area.
  params.filterByArea = True
  params.minArea = 500

  params.filterByConvexity = True
  params.minConvexity = 0.2
  detector = cv2.SimpleBlobDetector_create(params)
  # Detect blobs.
  keypoints = detector.detect(img)
  
  # Draw detected blobs as red circles.
  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
  im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  
  # Show keypoints
  #cv2_imshow("Keypoints", im_with_keypoints)
  return keypoints, im_with_keypoints

def process_contour(contour):
    MAJOR_DEFECT_THRESHOLD = 2.0
    
    ellipses = []
    
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull_idx)
    
    intersections = []
    for i,defect in enumerate(np.squeeze(defects, 1)):
        _, _, far_idx, far_dist = defect
        real_far_dist = far_dist / 256.0
        if real_far_dist >= MAJOR_DEFECT_THRESHOLD:
            intersections.append(far_idx)
    
    if len(intersections) == 0:
        print("One ellipse")
        ellipses = [cv2.fitEllipse(contour)]
    else:
      segments = []
      for i in range(len(intersections)-1):
        segments.append(contour[intersections[i]:intersections[i+1]+1])
      segments.append(np.vstack([contour[intersections[-1]:],contour[:intersections[0]+1]]))
      ellipses = [cv2.fitEllipse(c) for c in segments]
        
    return ellipses

circles = []

def distToClosestCircle(point):
  dist = 10000
  for c in circles:
    new_dist = math.sqrt((point[0] - c[0][0])**2 + (point[1] - c[0][1])**2)
    if new_dist < dist:
      dist = new_dist

  return dist

print(cv2.__version__)
vid = cv2.VideoCapture(0)
#for i in range(6):
while True:
    #path_to_img = 'Metal/Metal_' + str(i+1) + '.jpg'
    #image = cv2.imread(path_to_img)
    ret, image = vid.read()
    # show the image, provide window name first
    grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(grayscale, (3, 3), 0)
    #cv2_imshow(img_blur)
    adjusted = cv2.convertScaleAbs(img_blur, alpha=5.5, beta=10)
    adjusted = cv2.GaussianBlur(adjusted, (3, 3), 0)
    #cv2_imshow(adjusted)
    img_edge = edgeDetector(adjusted,algo_edgedetect='canny')

    contours, hierarchy = cv2.findContours(image=img_edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    image_copy = image.copy()
    circles = []
    for c in contours:

        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        x,y,w,h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        if hull_area > 0 and aspect_ratio > 0.3 and aspect_ratio < 3:
            solidity = float(area)/hull_area
            if hull_area > 100 and solidity > -0.1:
                #cv2.drawContours(image=image_copy, contours=c, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                ellipse = cv2.fitEllipse(c)
                ellipse_area = 3.1416 * ellipse[1][0] * ellipse[1][1]
                if distToClosestCircle(ellipse[0]) > 30:
                    circles.append(ellipse)
                    #cv2.ellipse(image_copy,cv2.fitEllipse(c), (0,255,0), 3)
                    print(ellipse_area)

                    #small ellipse
                    if ellipse_area < 1900:
                        if ellipse_area < 1500:
                            cv2.ellipse(image_copy,cv2.fitEllipse(c), (0,0,255), 3)
                        else:
                            cv2.ellipse(image_copy,cv2.fitEllipse(c), (0,255,0), 3)
                    elif ellipse_area < 2500:
                        cv2.ellipse(image_copy,cv2.fitEllipse(c), (255,0,0), 3)
                    elif ellipse_area < 4600 and aspect_ratio < 1:
                        if ellipse_area < 3500:
                            cv2.ellipse(image_copy,cv2.fitEllipse(c), (0,0,255), 3)
                        else:
                            cv2.ellipse(image_copy,cv2.fitEllipse(c), (0,255,0), 3)
                    else:
                        cv2.ellipse(image_copy,cv2.fitEllipse(c), (255,0,0), 3)
    cv2.imshow("output", image_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows() 
    # see the results
    #cv2.imshow(str(i),image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
#cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
#cv2_imshow(image)
#1200 -> small uncovered
# 1800 -> small covered


#2900 ~ 3300 -> big uncovered
#3750 ~ 4600 -> big covered

