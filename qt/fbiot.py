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
import fbchat
from numpy import array
from collections import Counter
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

class QtCapture(QtGui.QWidget):
    def loadtensorflow(self):
        self.sess = tf.Session()
        new_saver = tf.train.import_meta_graph('../Model/NIN-Model-0504-2.meta')
        new_saver.restore(self.sess, tf.train.latest_checkpoint('../Model/./'))
        self.all_vars = tf.trainable_variables()
        summary_writer = tf.summary.FileWriter('/tmp/rgbcnntest', self.sess.graph)
        #summary_writer = tf.train.SummaryWriter('/tmp/rgbcnntest', sess.graph)
        self.pred = tf.get_default_graph().get_tensor_by_name("pred/Reshape:0")
        self.keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
        self.px = tf.get_default_graph().get_tensor_by_name("placeholder_x:0")
        #print [n.name for n in tf.get_default_graph().as_graph_def().node if "pred" in n.name]
        return

    def loadanswer(self):
        with open("labels.p", "rb") as fp:
                self.labelname = pickle.load(fp)
        print "Sucessful Load Model"
        return

    def get_pic(self, frame):
        frame = Image.fromarray(frame, 'RGB')
        frame = frame.resize((60,50), Image.BILINEAR)
        flat_arr = np.array(frame)
        return flat_arr

    def __init__(self, *args):
        super(QtGui.QWidget, self).__init__()
	self.setStyleSheet('font-size: 20pt')
	self.loadtensorflow()
	self.loadanswer()
        self.fps = 24
	self.counter = 25
	self.righttime_ans = np.zeros(10)
	self.detect_postion = np.zeros((3,4))
	self.pout = [0, 0, 0]
	self.seq = np.zeros(200000)
	self.pri = []
	self.client = fbchat.Client("scure.le.1", sys.argv[1])
        self.cap = cv2.VideoCapture(0)
	self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1000)
	self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 600)

        self.video_frame = QtGui.QLabel()
        self.detect_frame = QtGui.QLabel()
        lay0 = QtGui.QHBoxLayout()
        lay0.setMargin(10)
        lay0.addWidget(self.video_frame)
        lay0.addWidget(self.detect_frame)

	self.word = QLabel()
	self.word.setText("CNN Detect Sign Pose: ")
	self.bu = QLabel()
	self.bu.setText("Ready to estimation...")
	
	buf = QtGui.QHBoxLayout()
	buf.addWidget(self.word)
	buf.addWidget(self.bu)

	lay = QVBoxLayout()
	lay.addLayout(lay0)
	lay.addLayout(buf)

        self.setLayout(lay)

    def setFPS(self, fps):
        self.fps = fps
    
    def DetectObject(self, hsv):
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
        ret, frame = self.cap.read()
	rightframe = np.copy(frame)
	hsv = np.copy(frame)
	righthand = np.zeros(4)
        lefthand = np.zeros(4)
	right_ans = -1
        # My webcam yields frames in BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	for detect in range(3):
	    x,y,w,h = self.DetectObject(hsv)
	    self.detect_postion[detect]=[x,y,w,h]
            hsv[int(y*0.7):int(y+1.3*h),int(x*0.7):int(x+1.3*w),:]=0
	self.detect_postion = self.detect_postion.astype(int)
	righthand, lefthand = self.classify3(righthand, lefthand)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
	hp = np.zeros((hsv.shape[0], hsv.shape[1], 3), np.uint8)
	if righthand[0]!=0:
	    cv2.circle(hp, (righthand[0]+(righthand[3]/2),righthand[1]+(righthand[2]/2)), 5, (0,0,255), -1)
	    if self.counter>25:
	        self.counter=0
	        rightframe = rightframe[righthand[1]:righthand[1]+righthand[3], righthand[0]:righthand[0]+righthand[2]]
	        rightflat_arr = self.get_pic(rightframe)
	        rightflat_arr = np.reshape(rightflat_arr,[-1,50,60,3])
                right_ans = self.sess.run(tf.argmax(self.pred,1), feed_dict={self.px: rightflat_arr, self.keep_prob: 1.})
	        for c in range(9):
	            self.righttime_ans[c+1]=self.righttime_ans[c]
                self.righttime_ans[0] = right_ans[0]
                righttime_ans_tmp = Counter(self.righttime_ans)
                right_ans = righttime_ans_tmp.most_common(1)[0]
	    self.counter = self.counter+1
	if lefthand[0]!=0:
	    cv2.circle(hp, (lefthand[0]+(lefthand[3]/2),lefthand[1]+(lefthand[2]/2)), 5, (0,255,0), -1)
        detimg = QtGui.QImage(hp, hp.shape[1], hp.shape[0], QtGui.QImage.Format_RGB888)
        detpix = QtGui.QPixmap.fromImage(detimg)
        self.video_frame.setPixmap(pix)
        self.detect_frame.setPixmap(detpix)
	if right_ans!=-1:
	    self.word.setText("Detect Sign Pose: "+str(self.labelname[int(right_ans[0])]))
	#self.bu.setText("You said:\na")
	if righthand[0]!=0 or lefthand[0]!=0:
	    if right_ans!=-1:
	        if Counter(self.righttime_ans).most_common(1)[0][1]==10:
	            for c in range(199999):
	                self.seq[c+1]=self.seq[c]
	            self.seq[0] = Counter(self.righttime_ans).most_common(1)[0][0]
	            if Counter(self.seq).most_common(1)[0][1]==200000:
	                self.pri = np.append(self.pri, str(self.labelname[int(Counter(self.seq).most_common(1)[0][0])]))
	                self.seq = np.zeros(200000)
	    self.bu.setText("You will send:\n"+str(self.pri))
	else:
	    if self.pri!=[]:
	        #self.seq = map(int, self.seq)
	        #self.seqlist.append(self.seq)
		#all_friends = self.client.getAllUsers()
		#friend = all_friends[0]
		#sent = self.client.send(friend.uid, str(self.labelname[int(self.seq[0])]))
		self.pri=[]
	#self.pout[1] = self.pout[1]+1
	self.counter = self.counter+1

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
