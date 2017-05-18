from __future__ import division
import numpy as np, random
from hmmlearn import hmm
from math import exp
import sys

def rand_array(row,col):
	return np.random.rand(row,col)

states = ["1", "2", "3", "4", "5"]
n_states = len(states)

observations = ["A", "B", "C", "D", "E"]
n_observations = len(observations)

start_probability = np.array([0.2, 0.2, 0.2, 0.3, 0.1])

transition_probability = np.array([[0.1, 0.3, 0.2, 0.3, 0.1],
				[0.2, 0.1, 0.2, 0.3, 0.2],
				[0.1, 0.1, 0.3, 0.2, 0.3],
				[0.2, 0.1, 0.4, 0.2, 0.1],
				[0.3, 0.1, 0.1, 0.3, 0.2]])

emission_probability = np.array([[0.1, 0.1, 0.1, 0.4, 0.3],
                                [0.2, 0.2, 0.2, 0.2, 0.2],
                                [0.3, 0.2, 0.1, 0.3, 0.1],
                                [0.2, 0.2, 0.3, 0.2, 0.1],
				[0.2, 0.2, 0.3, 0.2, 0.1]])

model = hmm.MultinomialHMM(n_components=n_states, n_iter=1000)
model.startprob=start_probability
model.transmat=transition_probability
model.emissionprob=emission_probability

#print model.transmat
#print model.emissionprob_.shape
#print model.startprob
#print

# predict a sequence of hidden states based on visible states                                                                                                                                                    
set1 = np.array(([0,1,1,2,3,1,0,1,2],
		[0,1,2,0,1,2],
		[0,3,2,0,0,1,2],
		[1,1,0,1,2,0,1],
		[1,2,0,0,1,2,2,0,1],
		[2,0,3,2,0,1,2,0],
		[2,0,1,2,0,1,2,0],
		[2,0,1,2,0],
		[2,0,1,2,0]))
for i in range(9):
	seq = np.array([set1[i]]).T
	model = model.fit(seq)


start_probability2 = rand_array(n_states, 1)

transition_probability2 = rand_array(n_states, n_states)

emission_probability2 = rand_array(n_states, n_observations)


bob_says = np.array([[0,0,1,2,0,2,2,1]]).T
logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
print "logprob", exp(logprob)
#print "Bob says:", ", ".join(map(lambda x: observations[x], bob_says.T[0]))
#print "Alice hears:", ", ".join(map(lambda x: states[x], alice_hears))
