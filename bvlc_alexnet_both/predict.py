import sys
import numpy as np
sys.path.append('/home/qinhf/caffe/python')
import caffe
import matplotlib.pyplot as plt
from scipy import misc
import scipy.io as sio

WEIGHTS_FILE = sys.argv[2]
DEPLOY_FILE = sys.argv[3]
MEAN_VALUE = 0.26448

#caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.array([MEAN_VALUE]))
#transformer.set_raw_scale('data', 255)

image_list = sys.argv[1]
pwd = sys.path[0]
print(pwd)
batch_size = net.blobs['data'].data.shape[0]
with open(image_list, 'r') as f:
    i = 0
    filenames = []
    res_txt = 'res_{}_{}.txt'.format(pwd.split('/')[-1], WEIGHTS_FILE.split('/')[-1].split('_')[-1][:-11])
    print(res_txt)
    with open(res_txt, 'w+') as f_out:
        for line in f.readlines():
            filename = line[:-1]
            filenames.append(filename)
            image = caffe.io.load_image(filename, False)
            #print(image)
            #misc.imsave('test.jpg', image)
            #plt.imshow('image')
            sio.savemat('test.mat', {'image':image} )

            transformed_image = transformer.preprocess('data', image)

            #print(transformed_image.shape)
            image = np.vstack((transformed_image, transformed_image, transformed_image))
            #print(image.shape)
            #exit(0)
            net.blobs['data'].data[i, ...] = transformed_image
            i += 1

            if i == batch_size:
                output = net.forward()
                freqs = output['pred']
                params = []
                for filename,  params in zip(filenames, freqs):
                    print('Predicted frequencies for {} is {}'.format(filename, params))
                    file_name_list = filename.split('/')
                    file_num = file_name_list[-1][:-4]
                    f_out.write('{}\t'.format(file_num))
                    #str_params = []
                    for item in range(0, len(params)):
                        f_out.write('{}\t'.format(params[item]))
                    f_out.write('\n')
                i = 0
                filenames = []
