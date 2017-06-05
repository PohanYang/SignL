import numpy as np
import pickle
import sys
from pomegranate import *

print "start testing" 

with open("hmm_data/model/hello.p", "rb") as fp:
    model1 = pickle.load(fp)
with open("hmm_data/model/ok.p", "rb") as fp:
    model2 = pickle.load(fp)
with open("hmm_data/model/yes.p", "rb") as fp:
    model3 = pickle.load(fp)

test = ['t_hello', 't_ok', 't_yes']

for n in test:
	print "Now testing ", n
	with open("hmm_data/"+n+".npy", "rb") as fp:
		loadmodel = pickle.load(fp)
	for m in range(len(loadmodel)):
		if model1.log_probability(map(str,loadmodel[m]))>model2.log_probability(map(str,loadmodel[m])) and model1.log_probability(map(str,loadmodel[m]))>model3.log_probability(map(str,loadmodel[m])):
			print "hello"
		if model2.log_probability(map(str,loadmodel[m]))>model1.log_probability(map(str,loadmodel[m])) and model2.log_probability(map(str,loadmodel[m]))>model3.log_probability(map(str,loadmodel[m])):
			print "ok"
		if model3.log_probability(map(str,loadmodel[m]))>model1.log_probability(map(str,loadmodel[m])) and model3.log_probability(map(str,loadmodel[m]))>model2.log_probability(map(str,loadmodel[m])):
			print "yes"


#print model.log_probability(map(str,))
#print model.log_probability(map(str,))
#print model.log_probability(map(str,))
