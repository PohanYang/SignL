import os
import glob
import sys
newn = sys.argv[2]
count = int(sys.argv[3])
path = sys.argv[1]
detail = sys.argv[4]
dirlist = ['1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','non']
for dlist in dirlist:
	for filename in glob.glob(os.path.join(path+'/'+str(dlist)+'/', '*.png')):
        	newname = newn+'/'+str(dlist)+'/'+detail+str(count)+'.png'
        	count = count + 1
        	os.rename(filename,newname)
