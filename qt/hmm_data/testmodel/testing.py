import numpy as np
import pickle
import sys
import operator
from pomegranate import *

print "Start testing"

def loadmodel():
	modellist = ['come']
	#modellist = ['come','hello', 'help', 'hold', 'hru', 'hungry', 'no', 'ok', 'sorry', 'soso', 'thx', 'unders', 'wtime', 'yes']
        hmmmodel=[]
        for model in modellist:
            with open(sys.argv[1]+model+".p", "rb") as fp:
                hmmmodel.append(pickle.load(fp))
        return hmmmodel

def answer(seq, hmmmodel):
	hmmpro = []
        modellist = ['come']
        #modellist = ['come','hello', 'help', 'hold', 'hru', 'hungry', 'no', 'ok', 'sorry', 'soso', 'thx', 'unders', 'wtime', 'yes']
        for hmmiter in range(len(hmmmodel)):
        	hmmpro.append(hmmmodel[hmmiter].log_probability(map(str, seq)))
        hmmindex, hmmvalue = max(enumerate(hmmpro), key=operator.itemgetter(1))
	return hmmindex

match = 0
nc = 0
hmmmodel = loadmodel()
test = ['test_come']
#test = ['test_come','test_hello', 'test_help', 'test_hold', 'test_hru', 'test_hungry', 'test_no', 'test_ok', 'test_sorry', 'test_soso', 'test_thx', 'test_unders', 'test_wtime', 'test_yes']
for n in test:
	#print "Testing ", n ,", now."
	with open(n+".npy", "rb") as fp:
		testseq = pickle.load(fp)
	for m in range(len(testseq)):
		ans = answer(testseq[m], hmmmodel)
		if int(ans)==int(nc):
			match+=1
		else:
			print n, m
	nc+=1
print "Accuracy: ", match, "/ 290"
