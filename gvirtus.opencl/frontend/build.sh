#!/bin/bash

gcc -c -Wall -Werror -fpic Opencl_gv.cpp
gcc -shared -o libOpencl_gv.so Opencl_gv.o 

gcc -c -Wall -Werror -fpic Opencl_Platform.cpp
gcc -shared -o libOpencl_platf.so Opencl_Platform.o
gcc -L/home/0108001475/gvirtus-opencl/frontend/ -lOpencl_gv -lOpencl_platf hellocl_platf.c -o hellocl_platf.exe

