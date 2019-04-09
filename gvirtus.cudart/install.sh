#!/bin/bash

INSTALL_FOLDER=$1

cd util
bash util.sh ${INSTALL_FOLDER}
cd ..

cd backend
bash backend.sh ${INSTALL_FOLDER}
cd ..

cd frontend
bash frontend.sh ${INSTALL_FOLDER}
cd ..

echo
/bin/echo -e "\e[1;30;102;5mGVIRTUS CUDA RUNTIME MODULE INSTALLATION COMPLETE!\e[0m"
echo
echo
