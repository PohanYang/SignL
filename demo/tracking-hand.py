import cv2
import numpy
import numpy as np
import time
import os
import glob
import tensorflow as tf
import math
from numpy import array
from collections import Counter
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

#Open Camera object
cap = cv2.VideoCapture(0)

#Decrease frame size
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 600)

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

def classify3(detect_postion, show_frame):
    detect_postion = sorted(detect_postion,key=lambda l:l[0], reverse=True)
    detect_postion = array(detect_postion)
    if detect_postion[2][0]==0 and detect_postion[2][1]==0 and detect_postion[2][2]==0 and detect_postion[1][0]==0 and detect_postion[1][1]==0 and detect_postion[1][2]==0:
        return detect_postion, show_frame
    elif detect_postion[2][0]==0 and detect_postion[2][1]==0 and detect_postion[2][2]==0 and detect_postion[1][0]!=0 and detect_postion[1][1]!=0 and detect_postion[1][2]!=0:
        img = cv2.rectangle(show_frame,(detect_postion[1][0]+(detect_postion[1][2]/2),detect_postion[1][1]+(detect_postion[1][3]/2)),(detect_postion[1][0]+(detect_postion[1][2]/2)+5,detect_postion[1][1]+(detect_postion[1][3]/2)+5),(0,0,255),2)
        return detect_postion, show_frame
    else:
        img = cv2.rectangle(show_frame,(detect_postion[0][0]+(detect_postion[0][2]/2),detect_postion[0][1]+(detect_postion[0][3]/2)),(detect_postion[0][0]+(detect_postion[0][2]/2)+5,detect_postion[0][1]+(detect_postion[0][3]/2)+5),(255,0,0),2)
        #img = cv2.rectangle(show_frame,(detect_postion[1][0]+(detect_postion[1][2]/2),detect_postion[1][1]+(detect_postion[1][3]/2)),(detect_postion[1][0]+(detect_postion[1][2]/2)+5,detect_postion[1][1]+(detect_postion[1][3]/2)+5),(255,0,0),2)
        img = cv2.rectangle(show_frame,(detect_postion[2][0]+(detect_postion[2][2]/2),detect_postion[2][1]+(detect_postion[2][3]/2)),(detect_postion[2][0]+(detect_postion[2][2]/2)+5,detect_postion[2][1]+(detect_postion[2][3]/2)+5),(0,0,255),2)
        return detect_postion, show_frame

    #head_index = median(detect_postion)
    #    if ((detect_postion[rank][0]+(detect_postion[rank][2]/2))<(head_postion[0]+20) and (detect_postion[rank][0]+(detect_postion[rank][2]/2))>(head_postion[0])-20) and ((detect_postion[rank][1]+(detect_postion[rank][3]/2))<(head_postion[1]+20) and (detect_postion[rank][1]+(detect_postion[rank][3]/2))>(head_postion[1])-20):
            #print "head is: ", rank,  detect_postion[rank][0], detect_postion[rank][1], detect_postion[rank][2], detect_postion[rank][3]
    #        switch_list(detect_postion,rank, 0)
    #if detect_postion[1][0]!=0 and detect_postion[2][0]!=0 and detect_postion[1][0]>detect_postion[2][0]:
    #    switch_list(detect_postion, 1, 2)
    return detect_postion, show_frame

count = 0
head_count = 0
time_ans = np.zeros(10)
detect_postion = np.zeros((3,4))
head_postion = np.zeros(2)
odd = 10
while(1):
    # Capture frame-by-frame
    ret, frame = cap.read()
    show_frame = np.copy(frame)
    if odd==10:
        #odd=0
        for detect in range(3):
            x,y,w,h = DetectObject(frame)
            detect_postion[detect]=[x,y,w,h]
            frame[y*0.7:y+1.3*h,x*0.7:x+1.3*w,:]=0
        t_frame = np.copy(frame)
        #print detect_postion
        detect_postion = detect_postion.astype(int)
        #if head_count<130:
        #    show_frame, head_postion, head_count, head_lock, head_succ = head_init(show_frame,detect_postion, head_postion, head_count, head_lock, head_succ)
        #print "lock_head postion: ", head_postion
        #if head_lock==True and head_succ==True:
        #    detect_postion = classify3(detect_postion, head_postion)
        detect_postion, show_frame = classify3(detect_postion, show_frame)
        #for i in range(y,y+h):
        #    for j in range(x,x+w):
        #        if mask2[i][j]==0:
	#	    t_frame[i,j,:]=0
        frame[y:y+h,x:x+w]
        #if (x-0.5*w)<0 and y!=0:
        #    frame = frame[y-0.5*h:y+1.5*h,x:x+1.5*w]
        #elif (y-0.5*h)<0 and x!=0:
        #    frame = frame[y:y+1.5*h,x-0.5*w:x+1.5*w]
        #elif (x-0.5*w)<0 and (y-0.5*h)<0:
        #    frame = frame[y:y+1.5*h,x:x+1.5*w]
        #else:
        #    frame = frame[y-0.5*h:y+1.5*h,x-0.5*w:x+1.5*w]
        #cv2.imshow('frame', t_frame)
        img = Image.fromarray(frame, 'RGB')
        img = img.resize((50,60), Image.BILINEAR)
        arr = np.array(img)
        flat_arr = arr.ravel()
        flat_arr = flat_arr.reshape((1,9000))
        # Display the resulting frame
        cv2.imshow('frame', show_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if odd!=10:
        odd+=1
        time_ans_tmp = Counter(time_ans)
        ans = time_ans_tmp.most_common(1)[0]
        #draw_frame.text((3,show_frame.size[1]-85), str(int(ans[0])),(0,0,255), font=font)
        cv2.imshow('frame', show_frame)
    #print sess.run(tf.argmax(output_layer, 1), feed_dict={px: flat_arr})[len(sess.run(tf.argmax(output_layer, 1), feed_dict={px: flat_arr}))-1]
    ##############write data##########################
    #filename = 'pic'+str(count)+'.png'
    #count = count +1
    #cv2.imwrite(filename, frame) 
    ##################################################
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
