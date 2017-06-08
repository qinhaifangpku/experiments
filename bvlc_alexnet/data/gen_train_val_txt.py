#!/usr/bin/python
#-*- coding: utf-8 -*-
# qhf 2017-6-2
# list all files in the folder
import sys
import os
import re
import random

with open('train_whole.txt','w') as f_train_txt:
    img_dir = sys.argv[1]
    txt_dir = sys.argv[2]
    count = 0
    file_list = []
    file_list = os.listdir(img_dir)
    inst = []
    for fi in file_list:
        fi.strip('\n')
        count  +=  1
        img_name = os.path.join(img_dir, fi)
        txt_name = fi[:-4] + '.txt'
        print(txt_name)
        txt_name = os.path.join(txt_dir, txt_name)
        print(txt_name)
        print(img_name)
        param_list = []
        with open (txt_name, 'r') as params:
            for num in params:
                num.strip('\n')
                param_list.append(num[:6])
        parameters = "\t".join(param_list)
        #print(parameters)
        temp = img_name + '\t' + parameters
        inst.append(temp)

    random.shuffle(inst)
    for item in inst:
        f_train_txt.write('{}\n'.format(item))

    print(count)
