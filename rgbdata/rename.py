import os
import glob
import sys
newn = sys.argv[2]
count = int(sys.argv[3])
path = sys.argv[1]
for filename in glob.glob(os.path.join(path, '*.png')):
        newname = newn+str(count)+'.png'
        count = count + 1
        os.rename(filename,newname)
