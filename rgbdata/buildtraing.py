from PIL import Image
from progress.bar import Bar
import numpy as np
import sys
import glob
import os.path
import pickle
import cv2

if len(sys.argv)!=2:
	print "Please print dictionary like -> $python buildtraining.py path"
	sys.exit()

class signLClass:
        def __init__(self):
                self.data = [np.zeros(1),np.zeros(1)]
		self.testdata = [np.zeros(1),np.zeros(1)]
                self.labels = []
                self.testlabels = []
                self.first = True
                self.testfirst = True
        def add_image(self, image, label):
                if self.first == False:
                        tmp_image = self.data[1]
                        tmp_label = self.data[0]
                self.data[1] = image
                self.data[0] = label
                #self.data[1] = np.reshape(self.data[1], [-1, 50, 60, 3])
                #self.data[0] = self.data[0].reshape((1,kind))
                if self.first == False:
                        self.data[1] = np.vstack((tmp_image, self.data[1]))
                        self.data[0] = np.vstack((tmp_label, self.data[0]))
                self.first = False
        def add_test_image(self, image, label):
                if self.testfirst == False:
                        tmp_image = self.testdata[1]
                        tmp_label = self.testdata[0]
                self.testdata[1] = image
                self.testdata[0] = label
                #self.data[1] = np.reshape(self.data[1], [-1, 50, 60, 3])
                #self.data[0] = self.data[0].reshape((1,kind))
                if self.testfirst == False:
                        self.testdata[1] = np.vstack((tmp_image, self.testdata[1]))
                        self.testdata[0] = np.vstack((tmp_label, self.testdata[0]))
                self.testfirst = False
        def train(self, n, batch_size):
                return signL.data[1][n:n+batch_size]
        def label(self, n, batch_size):
                return signL.data[0][n:n+batch_size]

def load_class(signL, loadname):
        signL = pickle.load(open(loadname, "rb"))
        return signL

def save_class(classname, savename):
	pickle.dump(classname, open(savename, "wb"))

def load_class(signL, loadname):
	signL = pickle.load(open(loadname, "rb"))
	return signL

def init():
	train_name = raw_input('Input training data name: ')
	#input_fin = False
	dict_path = os.listdir(sys.argv[1])
	return train_name, dict_path

def get_pic(filename):
	img = Image.open(filename)
	img = img.resize((60,50,3), Image.BILINEAR)
	flat_arr = np.array(img)
	return flat_arr

def load_data(signL, train_name, dict_path):
	label_index = 0
	test_counter = 0
	first = True
	bar = Bar('Loading training data...', max = len(dict_path))
	for dic in dict_path:
		labelnum = np.zeros(len(dict_path), dtype=np.int)
		labelnum[label_index] += 1
		for filename in glob.glob(os.path.join(sys.argv[1]+dic, '*.png')):
			signImage = get_pic(filename)
			signImage = np.reshape(signImage, [-1,50,60,3])
			if test_counter%100==0:
				signL.add_test_image(signImage, labelnum)
				test_counter+=1
				continue
			signL.add_image(signImage, labelnum)
			test_counter+=1
		signL.labels.append(dic)
		label_index += 1
		bar.next()
	bar.finish()
	return signL
def my_shuffle(signL):
	bar = Bar('Shuffling...', max = int(signL.data[0].shape[0]))
	for ss in range(signL.data[0].shape[0]):
		changeint = np.random.randint(signL.data[0].shape[0], size=1)[0]
		signL.data[0][ss], signL.data[0][changeint] = signL.data[0][changeint], signL.data[0][ss]
		signL.data[1][ss], signL.data[1][changeint] = signL.data[1][changeint], signL.data[1][ss]
		bar.next()
	bar.finish()
	return signL

def main():
	train_name, dict_path = init()
	signL = signLClass()
	signL = load_data(signL, train_name, dict_path)
	signL = my_shuffle(signL)
	#np.random.shuffle(signL.data)
	save_class(signL, train_name+'.p')

if __name__ == "__main__":
    main()
