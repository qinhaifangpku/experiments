#!/usr/bin/env sh
CAFFE_BIN=/home/qinhf/caffe/build/tools/caffe

LOG="log/train_log.txt"

${CAFFE_BIN} train -solver=./config/solver.prototxt  \
    -weights=./model/VGG_ILSVRC_16_layers.caffemodel 2>&1|tee ${LOG}
