#!/bin/bash

g++ -I/usr/local/include -L/usr/local/lib/ \
    -g -o landmark landmark.cpp \
    -O3 -Wall -Wextra -pedantic\
    -Wno-long-long -Wno-unused-parameter -Wstrict-aliasing\
    -lopencv_core -lopencv_imgproc -lopencv_highgui\
    -lopencv_ml -lopencv_video -lopencv_features2d\
    -lopencv_calib3d -lopencv_objdetect -lopencv_contrib\
    -lopencv_legacy -lopencv_stitching