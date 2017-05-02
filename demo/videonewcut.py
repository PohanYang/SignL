import cv2
import numpy
import numpy as np
import time
import os
import glob
import tensorflow as tf
import math
import pickle
import sys
from numpy import array
from collections import Counter
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw



def nothing(x):
    pass

# Function to find angle between two vectors
def Angle(v1,v2):
    dot = np.dot(v1,v2)
    x_modulus = np.sqrt((v1*v1).sum())
    y_modulus = np.sqrt((v2*v2).sum())
    cos_angle = dot / x_modulus / y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle

# Function to find distance between two points in a list of lists
def FindDistance(A,B):
    return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2))

def printing_word(frame,word, font_size, x, y):
    frame = Image.fromarray(frame)    
    draw_frame= ImageDraw.Draw(frame)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf", font_size)
    draw_frame.text((x,y), str(word), (0,0,255), font=font)
    frame = np.array(frame)
    return frame

def DetectObject(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #Create a binary image with where white will be skin colors and rest is black
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

def switch_list(detect_postion, A, B):
    detect_postion[A], detect_postion[B] = detect_postion[B], detect_postion[A]
    return detect_postion
    
def classify3(detect_postion, righthand, lefthand, show_frame):
    detect_postion = sorted(detect_postion,key=lambda l:l[0], reverse=True)
    detect_postion = array(detect_postion)
    if detect_postion[2][0]==0 and detect_postion[2][1]==0 and detect_postion[2][2]==0 and detect_postion[1][0]==0 and detect_postion[1][1]==0 and detect_postion[1][2]==0:
        return detect_postion, righthand, lefthand, show_frame
    elif detect_postion[2][0]==0 and detect_postion[2][1]==0 and detect_postion[2][2]==0 and detect_postion[1][0]!=0 and detect_postion[1][1]!=0 and detect_postion[1][2]!=0:
        righthand = detect_postion[1]
        img = cv2.rectangle(show_frame,(detect_postion[1][0]+(detect_postion[1][2]/2),detect_postion[1][1]+(detect_postion[1][3]/2)),(detect_postion[1][0]+(detect_postion[1][2]/2)+5,detect_postion[1][1]+(detect_postion[1][3]/2)+5),(0,0,255),2)
        return detect_postion, righthand, lefthand, show_frame
    else:
        righthand = detect_postion[2]
        lefthand = detect_postion[0]
        img = cv2.rectangle(show_frame,(detect_postion[0][0]+(detect_postion[0][2]/2),detect_postion[0][1]+(detect_postion[0][3]/2)),(detect_postion[0][0]+(detect_postion[0][2]/2)+5,detect_postion[0][1]+(detect_postion[0][3]/2)+5),(255,0,0),2)
        img = cv2.rectangle(show_frame,(detect_postion[2][0]+(detect_postion[2][2]/2),detect_postion[2][1]+(detect_postion[2][3]/2)),(detect_postion[2][0]+(detect_postion[2][2]/2)+5,detect_postion[2][1]+(detect_postion[2][3]/2)+5),(0,0,255),2)
        return detect_postion, righthand, lefthand, show_frame

    return detect_postion, show_frame

		

def init():
	global count, head_count, righttime_ans, leftime_ans, detect_postion, head_postion, odd
	count = 0
	head_count = 0
	righttime_ans = np.zeros(10)
	lefttime_ans = np.zeros(10)
	detect_postion = np.zeros((3,4))
	head_postion = np.zeros(2)
	odd = 10
def loadtensorflow():
	global sess, all_vars
	sess = tf.Session()
	new_saver = tf.train.import_meta_graph('../Model/NIN-Model-0420.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('../Model/./'))
	all_vars = tf.trainable_variables()
	summary_writer = tf.summary.FileWriter('/tmp/rgbcnntest', sess.graph)
	#summary_writer = tf.train.SummaryWriter('/tmp/rgbcnntest', sess.graph)
	pred = tf.get_default_graph().get_tensor_by_name("pred/Reshape:0")
	keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
	px = tf.get_default_graph().get_tensor_by_name("placeholder_x:0")
	#print [n.name for n in tf.get_default_graph().as_graph_def().node if "pred" in n.name]
	return pred, keep_prob, px

def loadanswer():
	with open("labels.p", "rb") as fp:
		labelname = pickle.load(fp)
	print "Sucessful Load Model"
	return labelname
def webcam_init():
	#Open Camera object
	cap = cv2.VideoCapture(sys.argv[1])
	#Decrease frame size
	cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1000)
	cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 600)
	print "Sucessful Webcam"
	return cap

def get_pic(frame):
	frame = Image.fromarray(frame, 'RGB')
        frame = frame.resize((60,50), Image.BILINEAR)
        flat_arr = np.array(frame)
        return flat_arr

def output(cap, pred, keep_prob, px, labelname):
	while(1):
		global count, head_count, righttime_ans, leftime_ans, detect_postion, head_postion, odd
		righthand = np.zeros(4)
		lefthand = np.zeros(4)
		ret, frame = cap.read()
		if ret==0 :
			continue
		rightframe = np.copy(frame)
		leftframe = np.copy(frame)
		show_frame = np.copy(frame)
		if odd == 10:
			odd=0
			for detect in range(3):
				x,y,w,h = DetectObject(frame)
				detect_postion[detect]=[x,y,w,h]
				frame[int(y*0.7):int(y+1.3*h),int(x*0.7):int(x+1.3*w),:]=0
			t_frame = np.copy(frame)
			detect_postion = detect_postion.astype(int)
			detect_postion, righthand, lefthand, show_frame = classify3(detect_postion, righthand, lefthand, show_frame)
			#print detect_postion
			if righthand[0]!=0:
				rightframe = rightframe[righthand[1]:righthand[1]+righthand[3], righthand[0]:righthand[0]+righthand[2]]
				rightflat_arr = get_pic(rightframe)
            			rightflat_arr = np.reshape(rightflat_arr,[-1,50,60,3])
            			right_ans = sess.run(tf.argmax(pred,1), feed_dict={px: rightflat_arr, keep_prob: 1.})
            			for c in range(9):
                			righttime_ans[c+1]=righttime_ans[c]
            			righttime_ans[0] = right_ans[0]
            			righttime_ans_tmp = Counter(righttime_ans)
            			right_ans = righttime_ans_tmp.most_common(1)[0]
            			print right_ans[0]
            			show_frame = printing_word(show_frame, "Your Hand Sign is: "+str(labelname[int(right_ans[0])]), 60, 4, 400)
        		if False:
				leftframe = leftframe[lefthand[1]:lefthand[1]+lefthand[3], lefthand[0]:lefthand[0]+lefthand[2]]
            			leftimg = Image.fromarray(leftframe, 'RGB')
            			leftimg = leftimg.resize((60,50), Image.BILINEAR)
            			leftflat_arr = np.array(leftimg)
            			leftflat_arr = np.reshape(leftflat_arr,[-1,50,60,3])
            			left_ans = sess.run(tf.argmax(pred,1), feed_dict={px: leftflat_arr, keep_prob: 1.})
            			for c in range(9):
                			lefttime_ans[c+1]=lefttime_ans[c]
            			lefttime_ans[0] = left_ans[0]
            			lefttime_ans_tmp = Counter(lefttime_ans)
            			left_ans = lefttime_ans_tmp.most_common(1)[0]
        		cv2.imshow('frame', show_frame)
        		if cv2.waitKey(1) & 0xFF == ord('q'):
            			break
		if odd!=10:
        		odd+=1
        		#righttime_ans_tmp = Counter(righttime_ans)
        		#right_ans = righttime_ans_tmp.most_common(1)[0]
        		show_frame = printing_word(show_frame, "Your Hand Sign is: "+str(labelname[int(right_ans[0])]), 60, 4, 400)
        		cv2.imshow('frame', show_frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
            			break
	cap.release()
	cv2.destroyAllWindows()

def main():
	pred, keep_prob, px = loadtensorflow()
	labelname = loadanswer()
	cap = webcam_init()
	init()
	print "Initialize done"
	output(cap, pred, keep_prob, px, labelname)

if __name__ == "__main__":
	main()
