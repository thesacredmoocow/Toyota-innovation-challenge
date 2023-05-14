import os
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import matplotlib.pyplot as plt
vid = cv2.VideoCapture(0)

has_depth = True
try:
    openni2.initialize()
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()

    depth_stream.start()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat= c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX= 640, resolutionY= 480, fps= 30))
except Exception:
    has_depth = False
refPt= []
selecting= False

# Initial OpenCVWindow Functions
#cv2.namedWindow("Depth Image")

while True: 
    # Grab a new depth frame
    if has_depth:
        frame= depth_stream.read_frame()
        #print(frame)
        #cv2.imshow("Depth Image", frame)
        frame_data= frame.get_buffer_as_uint16()
        
        # Put the depth frame into a numpy array and reshape it
        img= np.frombuffer(frame_data, dtype=np.uint16)
        #print(img)
        img.shape = (1, 480, 640)
        img= np.concatenate((img, img, img), axis=0)
        img= np.swapaxes(img, 0, 2)
        img= np.swapaxes(img, 0, 1)
        #img = img // int(32768/256)
        #print(img[100,:,0])
        #if len(refPt) > 1:
        #    img= img.copy()
        #    cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
            
        # Display the reshaped depth frame using OpenCV
        cv2.imshow("Depth Image", img)
        #cv2.imshow('frame', frame)
      
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ret, frame = vid.read()
  
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()