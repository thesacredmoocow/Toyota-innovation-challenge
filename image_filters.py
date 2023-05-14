import cv2
import numpy as np
import math

def houghLineDetector(img):
    
    img_edge = cv2.Canny(img,220,255)
    lines = cv2.HoughLinesP(img,0.1,np.pi/180,5,minLineLength=40,maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        return lines, img
    return None, None

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
        scharr_x = cv2.Scharr(img,-1,0,1)
        scharr_y = cv2.Scharr(img,-1,1,0)
        scharr_x_abs = np.uint8(np.absolute(scharr_x)) 
        scharr_y_abs = np.uint8(np.absolute(scharr_y)) 

        img_edge = cv2.bitwise_or(scharr_y_abs,scharr_x_abs) 
    return img_edge

def detectColorObjects(img,find_color=None, lowsat = 90, lowval = 50):
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    if find_color == 'red':
        Red1Lower = np.array([0,lowsat,lowval])
        Red1Upper = np.array([10,255,255])
        #(170-180)
        Red2Lower = np.array([170,lowsat,lowval])
        Red2Upper = np.array([180,255,255])
        img_mask1 = cv2.inRange(img_hsv,Red1Lower,Red1Upper)
        img_mask2 = cv2.inRange(img_hsv,Red2Lower,Red2Upper)
        img_mask = img_mask1+img_mask2
        masked_out = cv2.bitwise_and(img,img,mask=img_mask)
    if find_color == 'green':
        GreenLower = np.array([40,100,100])
        GreenUpper = np.array([80,255,255])
        img_mask = cv2.inRange(img_hsv,GreenLower,GreenUpper)
        masked_out = cv2.bitwise_and(img,img,mask=img_mask)
    return img,img_mask

def getContourEllipseFit(contour):
    #return 1
    img = np.zeros((480, 720, 3), dtype = np.uint8)
    img2 = np.zeros((480, 720, 3), dtype = np.uint8)
    hull_idx = cv2.convexHull(contour)
    cv2.fillPoly(img, pts =[hull_idx], color=(255,255,255))
    if len(contour) < 5:
        return 0
    e = cv2.fitEllipse(contour)

    cv2.ellipse(img2, e, (255,255,255), -1)
    #cv2.imshow("contour", img)
    #cv2.imshow("e", img2)

    contourPixels = np.sum(img == 255)
    ellipsePixels = np.sum(img2 == 255)

    return min(float(contourPixels)/ellipsePixels, float(ellipsePixels)/contourPixels)

def distToClosestCircle(circles, point):
  dist = 10000
  for c in circles:
    new_dist = math.sqrt((point[0] - c[0][0])**2 + (point[1] - c[0][1])**2)
    if new_dist < dist:
      dist = new_dist

def filterContours(contours, hierarchy):
    
    out_c = []
    out_ellipse = []
    for index, c in enumerate(contours):
        approximations = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c) 
        aspect_ratio = float(w)/h
        #if getContourEllipseFit(c) < 0.5:
        #    continue
        try:
            bestEllipse = cv2.fitEllipse(c)
            circularity = bestEllipse[1][1] / bestEllipse[1][0]
            ellipse_area = 3.1416 * bestEllipse[1][0] * bestEllipse[1][1]
        except Exception:
            continue
        duplicate = False
        for ellipses in out_ellipse:
            if distToClosestCircle(out_ellipse, bestEllipse[0]) < 30:
                duplicate = True
        if duplicate:
            continue
        if len(approximations) < 10:
            continue
        elif w < 5 or h < 5:
            continue
        elif area < 20:
            continue
        elif w < 10 and h < 10:
            continue
        elif w > 500 or h > 500:
            continue
        elif circularity > 4:
            continue
        elif aspect_ratio < 0.2 or aspect_ratio > 5:
            continue
        elif ellipse_area > 0 and ellipse_area < 500:
            continue
        elif cv2.isContourConvex(c):
            continue
        else:
            out_c.append(c)
    return out_c


def process_contour(contour, image):
    
    MAJOR_DEFECT_THRESHOLD = 2
    
    ellipses = []
    
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    hull_idx[::-1].sort(axis=0)
    defects = cv2.convexityDefects(contour, hull_idx)
    
    intersections = []
    for i,defect in enumerate(np.squeeze(defects, 1)):
        _, _, far_idx, far_dist = defect
        real_far_dist = far_dist / 256.0
        if real_far_dist >= MAJOR_DEFECT_THRESHOLD:
            intersections.append(far_idx)
            far = tuple(contour[far_idx][0])
            cv2.circle(image,far,2,[0,0,255],2)
    
    if len(intersections) == 0:
        #print("One ellipse")
        ellipses = [cv2.fitEllipse(contour)]
    else:
      segments = []
      for i in range(len(intersections)-1):
        segments.append(contour[intersections[i]:intersections[i+1]+1])
      segments.append(np.vstack([contour[intersections[-1]:],contour[:intersections[0]+1]]))
      for c in segments:
          if len(c) > 5:
            ellipses.append(cv2.fitEllipse(c))
        
    return ellipses, image


def filter_ellipse(ellipse, avgsize=0):
    ellipse_area = 3.1416 * ellipse[1][0] * ellipse[1][1]
    if ellipse[1][0] == 0:
        circularity = 10000
    else:
        circularity = ellipse[1][1] / ellipse[1][0]
    #print(ellipse)
    if avgsize > 0 and (ellipse_area < avgsize / 4 or ellipse_area > avgsize * 4):
        return False
    if ellipse_area < 500 or ellipse_area > 800000:
        return False
    if circularity > 4:
        return False
    return True

def get_line_angle(firstline, secondline):
    firstslope = math.atan2(firstline[0][3]-firstline[0][1], firstline[0][2]-firstline[0][0])
    secondslope = math.atan2(secondline[0][3]-secondline[0][1], secondline[0][2]-secondline[0][0])
    return math.abs(firstslope - secondslope)

def get_line_dist(p1, p2, p3):
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

def lines_identical(l1, l2):
    return (l1[0][0] == l2[0][0]) and (l1[0][1] == l2[0][1]) and (l1[0][2] == l2[0][2]) and (l1[0][3] == l2[0][3])

def merge_lines(lines):
    threshold = 10
    outputlines = []
    changed = True
    while changed:
        outputlines.clear()
        changed = False
        for lineA in lines:
            for lineB in lines:
                if not lines_identical(lineA, lineB):
                    if get_line_dist(np.array([lineA[0][0], lineA[0][1]]), np.array([lineA[0][2], lineA[0][3]]), np.array([lineB[0][0], lineB[0][1]])) < threshold:
                        if get_line_dist(np.array([lineA[0][0], lineA[0][1]]), np.array([lineA[0][2], lineA[0][3]]), np.array([lineB[0][2], lineB[0][3]])) < threshold:
                            lineA[0][2] = lineB[0][2]
                            lineA[0][3] = lineB[0][3]
                            outputlines.append(lineA)
                            changed = True
        lines = outputlines
    return lines

