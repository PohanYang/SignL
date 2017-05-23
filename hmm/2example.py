from __future__ import division
import numpy as np, random
from hmmlearn import hmm
from math import exp
import sys
import pickle

def rand_array(row,col):
	return np.random.rand(row,col)

with open("../qt/labels.p", "rb") as fp:
	states = pickle.load(fp)
n_states = len(states)

observations = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10',
		'b1','b2','b3','b4','b5','b6','b7','b8','b9','b10',
		'c1','c2','c3','c4','c5','c6','c7','c8','c9','c10',
		'd1','d2','d3','d4','d5','d6','d7']
n_observations = len(observations)

#start_probability = np.array([0.2, 0.2, 0.2, 0.3, 0.1])
start_probability = rand_array(n_states, 1)

transition_probability = rand_array(n_states, n_states)

emission_probability = rand_array(n_states, n_observations)

#transition_probability = np.array([[0.1, 0.3, 0.2, 0.3, 0.1],
#				[0.2, 0.1, 0.2, 0.3, 0.2],
#				[0.1, 0.1, 0.3, 0.2, 0.3],
#				[0.2, 0.1, 0.4, 0.2, 0.1],
#				[0.3, 0.1, 0.1, 0.3, 0.2]])

#emission_probability = np.array([[0.1, 0.1, 0.1, 0.4, 0.3],
#                                [0.2, 0.2, 0.2, 0.2, 0.2],
#                                [0.3, 0.2, 0.1, 0.3, 0.1],
#                                [0.2, 0.2, 0.3, 0.2, 0.1],
#				[0.2, 0.2, 0.3, 0.2, 0.1]])

model = hmm.MultinomialHMM(n_components=n_states, n_iter=10)
model.startprob=start_probability
model.transmat=transition_probability
model.emissionprob=emission_probability

#print model.transmat
#print model.emissionprob_.shape
#print model.startprob
#print

# predict a sequence of hidden states based on visible states                                                                                                                                                    
with open("../qt/matt.npy", "rb") as fp:
	set1 = pickle.load(fp)
add_first = list(range(37))
set1.append(add_first)
set1[0], set1[len(set1)-1] = set1[len(set1)-1], set1[0]
print set1

for i in range(len(set1)):
	print i
	seq = np.array([set1[i]]).T
	model = model.fit(seq)



#bob_says = np.array([[0,1,3,2,1]]).T
#logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
#print "logprob", exp(logprob)
##print "Bob says:", ", ".join(map(lambda x: observations[x], bob_says.T[0]))
##print "Alice hears:", ", ".join(map(lambda x: states[x], alice_hears))
