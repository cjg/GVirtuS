#!/bin/bash

INSTALL_FOLDER=$1

rsync -avR etc/gvirtus.properties $INSTALL_FOLDER

cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=${INSTALL_FOLDER} \
      -G "Unix Makefiles" -j 4 \
      . \
      --graphviz=.graphviz/gvirtus.dot
make
make install

dot -T pdf .graphviz/gvirtus.dot -o .graphviz/gvirtus.pdf

echo
/bin/echo -e "\e[1;30;102mGVIRTUS MODULE INSTALLATION COMPLETE!\e[0m"
echo
echo
