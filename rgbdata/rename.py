import os
import glob
count = 0
path = '5'
for filename in glob.glob(os.path.join(path, '*.png')):
        newname = 'training/5_'+str(count)+'.png'
        count = count + 1
        os.rename(filename,newname)
