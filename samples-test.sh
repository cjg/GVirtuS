#! /bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage $0 NVIDIA_SAMPLES_PATH"
    exit -1
fi

base_dir=$1

cd $base_dir
sample_type_paths=`ls -d ${base_dir}/?_*`
for sample_type_path in $sample_type_paths
do
  sample_paths=`ls -d ${sample_type_path}/*`
  for sample_path in $sample_paths
  do
    exe=$sample_path/`basename $sample_path`
    echo "-----------------------------------------"
    echo "Running: $exe"
    $exe >/dev/null
    if [ $? -eq 0 ]; then
      echo OK
    else
      echo FAIL
    fi
    echo "-----------------------------------------"
  done
done

