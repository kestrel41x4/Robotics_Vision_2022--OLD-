import numpy as np
import cv2 as cv

# Chessboard Params
xCam=9
yCam=13

# Image Capture Location
cap = cv.VideoCapture(1)

full = None
count = 0
nameCount = 0

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((xCam*yCam,3), np.float32)
objp[:,:2] = np.mgrid[0:xCam,0:yCam].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

while True:
    initRet, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (yCam,xCam), None)
    # If found, add object points, image points (after refining them)
    if ret == True and count > 10:
        cv.imwrite(str(nameCount)+'.jpg', img)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (yCam,xCam), corners2, ret)
        nameCount += 1
        count = 0

    cv.imshow('img', img)
    count += 1

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    
initRet, img = cap.read()
cap.release()
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(mtx)

h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

## undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv.imwrite('/images/calibresult.png', dst)
cv.waitKey(0)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )