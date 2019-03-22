#!/bin/bash

if [ $# -ne 1 ]; then
  echo usage: $0 "<path-of-installation-folder>"
  exit 1
fi

INSTALL_FOLDER=$1
echo $INSTALL_FOLDER

rsync -av gvirtus.properties $INSTALL_FOLDER/etc/
