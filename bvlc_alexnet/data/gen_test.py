#!/usr/bin/python
#-*- coding: utf-8 -*-
# qhf 2017-6-2
# list all files in the folder
import sys
import os
import re
import random

with open('test.txt','w') as f_train_txt:
    img_dir = sys.argv[1]
    count = 0
    file_list = []
    file_list = os.listdir(img_dir)
    inst = []
    for fi in file_list:
        fi.strip('\n')
        count  +=  1
        img_name = os.path.join(img_dir, fi)
        f_train_txt.write('{}\n'.format(img_name))

    print(count)
