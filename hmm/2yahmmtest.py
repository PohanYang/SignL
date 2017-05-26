import numpy as np
import pickle
from pomegranate import *


with open("../qt/labels.p", "rb") as fp:
	labelname = pickle.load(fp)
#print labelname

disc = []
dic = {}
for labelnum in range(37):
	dic[str(labelnum)] = 0.27

d1 = DiscreteDistribution(dic)
d2 = DiscreteDistribution(dic)

s1 = State(d1, name='s1')
s2 = State(d1, name='s2')

hmm = HiddenMarkovModel()
hmm.add_states(s1,s2)
hmm.add_transition(hmm.start, s1, 0.5)
hmm.add_transition(hmm.start, s2, 0.5)
hmm.add_transition(s1, s1, 0.5)
hmm.add_transition(s1, s2, 0.5)
hmm.add_transition(s2, s1, 0.5)
hmm.add_transition(s2, s2, 0.5)
hmm.bake()

with open("../qt/test_abbie.npy", "rb") as fp:
    tabbie = pickle.load(fp)
with open("../qt/test_matt.npy", "rb") as fp:
    tmatt = pickle.load(fp)

#print hmm.log_probability(map(str,tabbie[0]))
#print hmm.log_probability(map(str,tmatt[0]))

with open("../qt/abbie.npy", "rb") as fp:
	abbie = pickle.load(fp)
with open("../qt/matt.npy", "rb") as fp:
	matt = pickle.load(fp)

for n in range(len(matt)):
	matt[n]=map(str,matt[n])
matt.append(map(str, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]))
hmm.fit(matt)
print "Training Done" 

print hmm.log_probability(map(str,tabbie[0]))
print hmm.log_probability(map(str,tmatt[0]))
