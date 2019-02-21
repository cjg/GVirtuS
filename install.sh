#!/usr/bin/env bash

#This is just for first trial (FULL INCOMPLETE)
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" -j 4 ..
make && make install
cd ..
