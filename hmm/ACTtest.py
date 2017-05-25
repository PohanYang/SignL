from pomegranate import *
import numpy as np

d1 = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})
d2 = DiscreteDistribution({'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25})

s1 = State( d1, name='background' )
s2 = State( d2, name='CG island' )

hmm = HiddenMarkovModel()
hmm.add_states(s1, s2)
hmm.add_transition( hmm.start, s1, 0.5 )
hmm.add_transition( hmm.start, s2, 0.5 )
hmm.add_transition( s1, s1, 0.5 )
hmm.add_transition( s1, s2, 0.5 )
hmm.add_transition( s2, s1, 0.5 )
hmm.add_transition( s2, s2, 0.5 )
hmm.bake()

hmm.fit([list('CGCGGACTCGATCGTCGCCGT')])

seq = list('CGACTACTGACTACTCGCCGACGCGACTGCCGTCTATACTGCGCATACGGC')
print hmm.log_probability( seq )

hmm.fit([list('CAAGGCGATCTCGATCTGACCTTA')])

seq = list('CGACTACTGACTACTCGCCGACGCGACTGCCGTCTATACTGCGCATACGGC')
print hmm.log_probability( seq )
