import os
from tqdm import tqdm
import glob
import numpy as np

fea_dir = '../data/anet/fc6_feat_100rois'
features = glob.glob(os.path.join(fea_dir,'*.npy'))
for f in tqdm(features):
    try:
        a = np.load(f)
    except:
        with open('feature_error2', 'a') as aa:
            aa.writelines(f + '\n')
