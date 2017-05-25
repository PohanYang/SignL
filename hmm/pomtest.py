from pomegranate import *

d1 = DiscreteDistribution({'A' : 0.35, 'C' : 0.20, 'G' : 0.05, 'T' : 40})
d2 = DiscreteDistribution({'A' : 0.25, 'C' : 0.25, 'G' : 0.25, 'T' : 25})
d3 = DiscreteDistribution({'A' : 0.10, 'C' : 0.40, 'G' : 0.40, 'T' : 10})

s1 = State(d1, name="s1")
s2 = State(d2, name="s2")
s3 = State(d3, name="s3")
model = HiddenMarkovModel()
model.add_states([s1, s2, s3])
model.add_transition(model.start, s1, 0.90)
model.add_transition(model.start, s2, 0.10)
model.add_transition(s1, s1, 0.80)
model.add_transition(s1, s2, 0.20)
model.add_transition(s2, s2, 0.90)
model.add_transition(s2, s3, 0.10)
model.add_transition(s3, s3, 0.70)
model.add_transition(s3, model.end, 0.30)
model.bake()

model.fit([list('CGTTAATTCGT')])
model.fit([list('CGCTATGAT')])

print model.log_probability(list('CGCTTTCGT'))
print model.log_probability(list('ACGACTATGAT'))
