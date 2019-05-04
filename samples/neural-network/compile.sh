#!/bin/bash

make clean;
make SRC_DIR=src EXE=nn_gpufs
make clean;
make SRC_DIR=src_cuda EXE=nn_cuda GPUFSLIBS="" 

