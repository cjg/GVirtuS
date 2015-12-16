#!/bin/bash

gcc -c -Wall -Werror -fpic OpenclHandler.cpp
gcc -shared -o libOpencl_handler.so OpenclHandler.o
gcc -c -Wall -Werror -fpic OpenclBackend.cpp
gcc -shared -o libOpencl_backend.so OpenclBackend.o
gcc -c -Wall -Werror -fpic OpenclHandler_Platform.cpp
#gcc -L/home/0108001475/gvirtus-opencl/backend -lOpencl_backend 


