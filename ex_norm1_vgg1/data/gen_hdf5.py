import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py

IMAGE_SIZE = (200, 200)
MEAN_VALUE = 0.26526

filename = sys.argv[1]
setname, ext = filename.split('.')

with open(filename, 'r') as f:
    lines = f.readlines()

#np.random.shuffle(lines)

sample_size = len(lines)
params_size = 5
imgs = np.zeros((sample_size, 3,) + IMAGE_SIZE, dtype=np.float32)
freqs = np.zeros((sample_size, params_size), dtype=np.float32)
h5_filename = '{}.h5'.format(setname)
with h5py.File(h5_filename, 'w') as h:
    for i, line in enumerate(lines):
        temp_list= line[:-1].split('\t')
        #print(len(temp_list))
        image_name = temp_list[0]
        params = temp_list[1:]
        #print(image_name)
        #print(len(params))
        im = Image.open(image_name)
        #im = im.convert('RGB')
        img1 = np.fromiter(iter(im.getdata()), np.float32)
        img1.resize(1, 200, 200)
        img = np.vstack((img1, img1, img1))
        #MEAN_VALUE = img.mean()
        #print(MEAN_VALUE)
        #img = plt.imread(image_name)[:, :, 0].astype(np.float32)
        #img = img.reshape((1, )+img.shape)
        #MEAN_VALUE = img.mean()
        img -= MEAN_VALUE
        imgs[i] = img
        print(len(params))
        for ii in range(0, params_size):
            #print(ii)
            #npos = params[ii].index('.')
            #print(params[ii])
            #print(npos)
            #exit(0)
            freqs[i][ii] = float(params[ii])
        if (i+1) % 1000 == 0:
            print('Processed {} images!'.format(i+1))
    h.create_dataset('data', data=imgs)
    h.create_dataset('freq', data=freqs)

with open('{}_h5.txt'.format(setname), 'w') as f:
    f.write(h5_filename)
