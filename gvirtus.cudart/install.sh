#!/bin/bash

INSTALL_FOLDER=$1

cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=${INSTALL_FOLDER} \
      -G "Unix Makefiles" -j 4 \
      . \
      --graphviz=.graphviz/gvirtus.cudart
make
make install

echo
/bin/echo -e "\e[1;30;102mGVIRTUS CUDA RUNTIME MODULE INSTALLATION COMPLETE!\e[0m"
echo
echo
