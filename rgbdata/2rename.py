import os
import glob
import sys
newn = sys.argv[2]
count = int(sys.argv[3])
path = sys.argv[1]
detail = sys.argv[4]
dirlist = ['come','help_h2','howd1','howd2','hungry','likeb1','likeb2','me','nob1','nob2','non','sorry1','thanku1','time1','unders1','unders2','want1','want2','why1','why2','you']
for dlist in dirlist:
	for filename in glob.glob(os.path.join(path+'/'+str(dlist)+'/', '*.png')):
        	newname = newn+'/'+str(dlist)+'/'+detail+str(count)+'.png'
        	count = count + 1
        	os.rename(filename,newname)
