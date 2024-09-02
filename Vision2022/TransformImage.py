from cmath import tan
import cv2
import time
import numpy as np
import math
from pupil_apriltags import Detector
import glob
#from Constants import VisionConstants

at_detector = Detector(families='tag36h11',
                       nthreads=16,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

# Parameters gotten from passing in images to 
# AnalyzeDistortion.py
camera_parameters = [443.6319712,  # fx
                     391.50381628, # fy
                     959.49982957, # cx
                     539.49965467] # cy

cameraInUse = 0

# Setting up the camera feed
cap = cv2.VideoCapture(cameraInUse)

#TODO: Delete this when using videocapture 
# Setting up camera width and height
#if cameraInUse == 0:
#cap.set(4, 800)
#cap.set(4, 600)


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 10)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 10)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 10)
    return img
    
def drawBoxes(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    
    img = cv2.drawContours(img, [imgpts[:4]], -1 (0,255,0), -3)
    
    for i,j in zip(range(4), range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple (imgpts[j]), (255),3)
        
    img =cv2.drawContours(img, [imgpts[4:]], -1,(0,0,255),3)
    
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((24*17,3), np.float32)
objp[:,:2] = np.mgrid[0:24, 0:17].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axisBoxes = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3], [0,3,-3], [3,3,-3],[3,0,-3]])

for img in glob.glob("undistorted.png"):
    

# while(True):
#     ret, frame = cap.read()
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Tag size: 0.173m
#     tags = at_detector.detect(image, estimate_tag_pose=True, camera_params=camera_parameters, tag_size=0.173)

    
#     for tag in tags:
#         for p1, p2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
#             cv2.line(frame,
#                     (int(tag.corners[p1][0]), int(tag.corners[p1][1])),
#                     (int(tag.corners[p2][0]), int(tag.corners[p2][1])),
#                     (255, 0, 255), 2)
        
#         # Get X,Y value of center in np array form 
        
#         center = tag.center
        
        
        
        
        
        
    
#     # Display the resulting frame
#     cv2.imshow('Video Feed',frame)
#     # cv2.imshow('image',image)

#     # The time took to proccess the frame
#     endTime = time.monotonic()
#     # print(f"{endTime - startTime:.4f}")

#     # Waits for a user input to quit the application
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()