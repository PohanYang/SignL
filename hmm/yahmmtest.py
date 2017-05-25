import numpy as np
import pickle
from pomegranate import *


with open("../qt/labels.p", "rb") as fp:
	labelname = pickle.load(fp)
#print labelname

dic = {}
for labelnum in range(37):
	dic[str(labelnum)] = 0.27
d1 = DiscreteDistribution(dic)
cpt = []
for labelnum in range(37):
	for labelnum2 in range(37):
		cpt.append([str(labelnum), str(labelnum2), 0.50])
d2 = ConditionalProbabilityTable(cpt, [d1])
clf = MarkovChain([d1, d2])

with open("../qt/test_abbie.npy", "rb") as fp:
    tabbie = pickle.load(fp)
with open("../qt/test_matt.npy", "rb") as fp:
    tmatt = pickle.load(fp)

print clf.log_probability(map(str,tabbie[0]))
print clf.log_probability(map(str,tmatt[0]))

with open("../qt/abbie.npy", "rb") as fp:
	abbie = pickle.load(fp)
with open("../qt/matt.npy", "rb") as fp:
	matt = pickle.load(fp)

for n in range(10):
	abbie[n]=map(str,abbie[n])
clf.fit(abbie)
print "Training Done" 

print clf.log_probability(map(str,abbie[3]))
print clf.log_probability(map(str,tmatt[0]))
