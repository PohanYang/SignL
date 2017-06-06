import pickle

with open("excelfile.p", "rb") as fp:
	excelfile = pickle.load(fp)

for n in excelfile.split(' ,'):
	print n
