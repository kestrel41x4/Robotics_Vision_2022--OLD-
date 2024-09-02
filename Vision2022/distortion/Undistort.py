import numpy as np 
import cv2 as cv
import glob

chessboardSize = (24,17)
frameSize = ( 1440, 1080)

criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)


objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0 :chessboardSize[1]].T.reshape(-1,2)

objPoints = []
imgPoints = []

images = glob.glob('image*.jpg')

for image in images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtcolor(img, cv.COLOR_BGR2GRAY)
    
    ret , corners = cv.findChessboardCorners(gray, chessboardSize, None)
    
    if ret == True:
        objPoints.Append(objp)
        corners2 = cv.cornerSubPix(gray, corners , (11,11), (-1,-1), criteria)
        imgPoints.append(corners)
        
        cv.drawChessboardCorners(img,chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)
        

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)
print("CameraCalibrated: " , ret)
print("\nCamera Matrix: \n", cameraMatrix )
print("\nDistortion Parameters: ]n ", dist)
print("\nRotation Vectors: \n" , rvecs)
print("\ntranslation Vectors: \n" , tvecs)

np.savez("CameraParams", cameraMatrix = cameraMatrix, dist=dist, rvecs=rvecs, tvecs=tvecs)

img = cv.imread('image2.jpg')
h, w = img.shape[:2]
newCameraMatrix , roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist)
    