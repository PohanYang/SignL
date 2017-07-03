import numpy as np
import pickle
import sys
from pomegranate import *


with open("../demo/labels.p", "rb") as fp:
	labelname = pickle.load(fp)
#print labelname

disc = []
dic = {}
for labelnum in range(len(labelname)):
	dic[str(labelnum)] = float(1)/float(len(labelname))

d1 = DiscreteDistribution(dic)
d2 = DiscreteDistribution(dic)
d3 = DiscreteDistribution(dic)
d4 = DiscreteDistribution(dic)
d5 = DiscreteDistribution(dic)

s1 = State(d1, name='s1')
s2 = State(d1, name='s2')
s3 = State(d1, name='s3')
s4 = State(d1, name='s4')
s5 = State(d1, name='s5')

hmm = HiddenMarkovModel()
hmm.add_states(s1,s2,s3,s4,s5)
hmm.add_transition(hmm.start, s1, 0.5)
hmm.add_transition(hmm.start, s2, 0.5)
hmm.add_transition(hmm.start, s3, 0.5)
hmm.add_transition(hmm.start, s4, 0.5)
hmm.add_transition(hmm.start, s5, 0.5)
hmm.add_transition(s1, s1, 0.5)
hmm.add_transition(s1, s2, 0.5)
hmm.add_transition(s1, s3, 0.5)
hmm.add_transition(s1, s4, 0.5)
hmm.add_transition(s1, s5, 0.5)
hmm.add_transition(s2, s1, 0.5)
hmm.add_transition(s2, s2, 0.5)
hmm.add_transition(s2, s3, 0.5)
hmm.add_transition(s2, s4, 0.5)
hmm.add_transition(s2, s5, 0.5)
hmm.add_transition(s3, s1, 0.5)
hmm.add_transition(s3, s2, 0.5)
hmm.add_transition(s3, s3, 0.5)
hmm.add_transition(s3, s4, 0.5)
hmm.add_transition(s3, s5, 0.5)
hmm.add_transition(s4, s1, 0.5)
hmm.add_transition(s4, s2, 0.5)
hmm.add_transition(s4, s3, 0.5)
hmm.add_transition(s4, s4, 0.5)
hmm.add_transition(s4, s5, 0.5)
hmm.add_transition(s5, s1, 0.5)
hmm.add_transition(s5, s2, 0.5)
hmm.add_transition(s5, s3, 0.5)
hmm.add_transition(s5, s4, 0.5)
hmm.add_transition(s5, s5, 0.5)
hmm.bake()

with open("hmm_data/"+sys.argv[1]+".npy", "rb") as fp:
    model = pickle.load(fp)

#print hmm.log_probability(map(str,tabbie[0]))
#print hmm.log_probability(map(str,tmatt[0]))

#with open("../qt/abbie.npy", "rb") as fp:
#	abbie = pickle.load(fp)
#with open("../qt/matt.npy", "rb") as fp:
#	matt = pickle.load(fp)

print len(model)
model = model[0:len(model)-5]
for n in range(len(model)):
	model[n]=map(str,model[n])
model.append(map(str, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]))

hmm.fit(model)
print "Training Done" 

with open("hmm_data/model/"+sys.argv[1]+".p", "wb") as fp:
    pickle.dump(hmm, fp)

#print hmm.log_probability(map(str,yes[5]))
