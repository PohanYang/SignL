import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
import cv2
import numpy
import numpy as np
import time
import os
import glob
import tensorflow as tf
import math
import pickle
from numpy import array
from collections import Counter
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

class QtCapture(QtGui.QWidget):
    def __init__(self, *args):
        super(QtGui.QWidget, self).__init__()

        self.fps = 24
        self.cap = cv2.VideoCapture(0)
	self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1000)
	self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 600)

        self.video_frame = QtGui.QLabel()
        self.detect_frame = QtGui.QLabel()
        lay0 = QtGui.QHBoxLayout()
        lay0.setMargin(10)
        lay0.addWidget(self.video_frame)
        lay0.addWidget(self.detect_frame)

	buf = QtGui.QHBoxLayout()
	self.word = QLabel()
	self.word.setText("Sign Language Buffer: ")
	buf.addWidget(self.word)

	lay = QVBoxLayout()
	lay.addLayout(lay0)
	lay.addLayout(buf)

        self.setLayout(lay)

    def setFPS(self, fps):
        self.fps = fps

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
	hsv = np.copy(frame)
        # My webcam yields frames in BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        detimg = QtGui.QImage(hsv, hsv.shape[1], hsv.shape[0], QtGui.QImage.Format_RGB888)
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

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    # ------ Modification ------ #
    def saveCapture(self):
        if self.capture:
            self.capture.capture()
    # ------ Modification ------ #



if __name__ == '__main__':

    import sys
    app = QtGui.QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())
