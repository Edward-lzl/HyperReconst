import numpy as np
import os
data_path = '/home/bingxing2/ailab/group/ai4earth/data/weather-5k/WEATHER-5K/hrrr'

file_path = os.path.join(data_path,'train_data_hrrr_r1.npy')
data = np.load(file_path)
import pdb
pdb.set_trace()