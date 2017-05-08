import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
import cv2
import numpy
import numpy as np
import time
import os
import glob
import math
import pickle
from numpy import array
from collections import Counter
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

class QtCapture(QtGui.QWidget):
#this is pyQt class, for UI.
    def __init__(self, *args): #initial UI box
        super(QtGui.QWidget, self).__init__()

        self.fps = 24
	self.detect_postion = np.zeros((3,4))
        self.cap = cv2.VideoCapture(0)
	self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1000)
	self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 600)

        self.video_frame = QtGui.QLabel()
        self.detect_frame = QtGui.QLabel()
        lay0 = QtGui.QHBoxLayout()
        lay0.setMargin(10)
        lay0.addWidget(self.video_frame)
        lay0.addWidget(self.detect_frame)

        self.setLayout(lay0)

    def setFPS(self, fps):
    #set FPS
        self.fps = fps
    
    def DetectObject(self, hsv):
    #this function is about detect hand on video frame
    #Input video frame (there I named "hsv"), and output is detect hand postions x,y,w(width) and h(height)
	hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
	mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
	blur = cv2.GaussianBlur(mask2,(5,5),0)
	#Kernel matrices for morphological transformation
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1)
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        median = cv2.medianBlur(dilation2,5)
        ret, thresh = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        ret,thresh = cv2.threshold(median,127,255,0)
        #Find contours of the filtered frame
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        max_area=100
        ci=0
        for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i
        #Largest area contour
        if len(contours)==0:
            return [0,0,0,0]
        cnts = contours[ci]
        x,y,w,h = cv2.boundingRect(cnts)
        return x,y,w,h

    def classify3(self, righthand, lefthand):
    #because somtimes will get wrong detection, this function is classify detect object.
    #Even classify right hand and left hand in a frame.
        self.detect_postion = sorted(self.detect_postion,key=lambda l:l[0], reverse=True)
        self.detect_postion = array(self.detect_postion)
	if self.detect_postion[2][0]==0 and self.detect_postion[2][1]==0 and self.detect_postion[2][2]==0 and self.detect_postion[1][0]==0 and self.detect_postion[1][1]==0 and self.detect_postion[1][2]==0:
	    return righthand, lefthand
	elif self.detect_postion[2][0]==0 and self.detect_postion[2][1]==0 and self.detect_postion[2][2]==0 and self.detect_postion[1][0]!=0 and self.detect_postion[1][1]!=0 and self.detect_postion[1][2]!=0:
            righthand = self.detect_postion[1]
            return righthand, lefthand
	else:
	    righthand = self.detect_postion[2]
            lefthand = self.detect_postion[0]
	    return righthand, lefthand
	

    def nextFrameSlot(self):
    #this function will input video frame in streaming
	righthand = np.zeros(4)
        lefthand = np.zeros(4)
        ret, frame = self.cap.read()
	hsv = np.copy(frame)
        # My webcam yields frames in BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	for detect in range(3):
	#in this loop we detect 3 times for increase accuracy in a video frame
	    x,y,w,h = self.DetectObject(hsv)
	    self.detect_postion[detect]=[x,y,w,h]
            hsv[int(y*0.7):int(y+1.3*h),int(x*0.7):int(x+1.3*w),:]=0
	self.detect_postion = self.detect_postion.astype(int)
	righthand, lefthand = self.classify3(righthand, lefthand)#resave right hand postion and left hand postions in parameters
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
	hp = np.zeros((hsv.shape[0], hsv.shape[1], 3), np.uint8)
	if righthand[0]!=0:
	    cv2.circle(hp, (righthand[0]+(righthand[3]/2),righthand[1]+(righthand[2]/2)), 5, (0,0,255), -1)
	    # this for draw a point where is right hand in black scene
	if lefthand[0]!=0:
	    cv2.circle(hp, (lefthand[0]+(lefthand[3]/2),lefthand[1]+(lefthand[2]/2)), 5, (0,255,0), -1)
	    # this for draw a point where is left hand in black scene
        detimg = QtGui.QImage(hp, hp.shape[1], hp.shape[0], QtGui.QImage.Format_RGB888)
        detpix = QtGui.QPixmap.fromImage(detimg)
        self.video_frame.setPixmap(pix)
        self.detect_frame.setPixmap(detpix)

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000./self.fps)

    def stop(self):
        self.timer.stop()

    # ------ Modification ------ #
    def capture(self):
        if not self.isCapturing:
            self.isCapturing = True
        else:
            self.isCapturing = False
    # ------ Modification ------ #

    def deleteLater(self):
        self.cap.release()
        super(QtGui.QWidget, self).deleteLater()


class ControlWindow(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.capture = None
	
	self.title = QLabel()
	self.title.setText("Sign Language Estimation")
	self.logolabel = QLabel()
	pixmap = QPixmap('NCTU.png')
	pixmap1 = pixmap.scaled(160,160)
	self.logolabel.setPixmap(pixmap1)
	vbox0 = QtGui.QHBoxLayout()
	vbox0.addWidget(self.title)
	vbox0.addWidget(self.logolabel)

        self.start_button = QtGui.QPushButton('Start')
        self.start_button.clicked.connect(self.startCapture)
	self.end_button = QtGui.QPushButton('Stop')
        vbox1 = QtGui.QHBoxLayout()
        vbox1.addWidget(self.start_button)
	vbox1.addWidget(self.end_button)

	vbox = QVBoxLayout(self)
	vbox.addLayout(vbox0)
	vbox.addLayout(vbox1)

        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        self.setGeometry(100,100,200,200)
        self.show()

    def startCapture(self):
        if not self.capture:
            self.capture = QtCapture(0)
	    self.end_button.clicked.connect(self.capture.stop)
            # self.capture.setFPS(1)
            self.capture.setParent(self)
            self.capture.setWindowFlags(QtCore.Qt.Tool)
        self.capture.start()
        self.capture.show()

if __name__ == '__main__':

    import sys
    app = QtGui.QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())
