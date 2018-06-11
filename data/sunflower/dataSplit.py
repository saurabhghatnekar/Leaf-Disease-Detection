import os
from os import makedirs
source1 = "/mnt/attic/DADA/data/sunflower/training"
dest11 = "/mnt/attic/DADA/data/sunflower/testing"
folders = os.listdir(source1)

import shutil
import numpy as np
import random
for f in folders:
    files = os.listdir(source1+'/'+f)
    print(files)
    makedirs(dest11+"/"+f)
    nums = random.sample(range(len(files)),4)
    print(nums)
    for n in nums:
        pass
        # shutil.move(source1 + '/'+ f+'/'+files[n], dest11 + '/'+ f+'/'+files[n])
