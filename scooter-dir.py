
# coding: utf-8

from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

def turnEstimation(pts, img):

    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]


    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    cntr = (int(mean[0,0]), int(mean[0,1]))

    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)

    angle = atan2(eigenvectors[0,1], eigenvectors[0,0])

    return angle

if __name__== "__main__":
    
    cap = cv.VideoCapture('/home/yogesh/Downloads/test2.avi')
    ret, frame = cap.read()
    rows, cols = frame.shape[:2]
    video = cv.VideoWriter('outpy.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (cols,rows))

    while(cap.isOpened()):

            ret, frame = cap.read()
            
            windowWidth = 700
            windowHeight = 200
            windowCol = int((cols - windowWidth) / 2)
            windowRow = int((rows - windowHeight) / 2)
            track_window = (windowCol, windowRow+150, windowWidth, windowHeight)
            
            src = frame[windowRow+190:rows, windowCol+100:cols]
            hsv_roi =  cv.cvtColor(src, cv.COLOR_BGR2HSV)

            if src is None:
                print('Could not open or find the image: ', args.input)
                exit(0)

            gray  = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            _, contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            c = max(contours, key = cv.contourArea)
            angle=turnEstimation(c, src)
            angleRad = angle*180/pi    
            x,y,w,h = cv.boundingRect(c)

            if (angleRad > -4) and (angleRad < 4):
                cv.putText(frame,'straight',(10,500), cv.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv.LINE_AA)
            elif (angleRad <= -4):
                cv.putText(frame,'left',(10,500), cv.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv.LINE_AA)
            elif angleRad >= 4:
                cv.putText(frame,'right',(10,500), cv.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv.LINE_AA)

            cv.imshow('output', frame)
            video.write(frame)
            k = cv.waitKey(60) & 0xff
            if k == 27:
                break

    cv.destroyAllWindows()
    video.release()

