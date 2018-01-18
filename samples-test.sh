#! /bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage $0 NVIDIA_SAMPLES_PATH"
    exit -1
fi

echo "TYPE;SAMPLE;RESULT;ERROR"

base_dir=$1

cd $base_dir
sample_type_paths=`ls -d ${base_dir}/?_*`
for sample_type_path in $sample_type_paths
do
  sample_paths=`ls -d ${sample_type_path}/*`
  sample_type=`basename $sample_type_path`
  for sample_path in $sample_paths
  do
    sample_exec=`basename $sample_path`
    cd $sample_path
    if [ -f $sample_exec ]; then
      rm /tmp/err.txt
      err=""
      ./$sample_exec >/dev/null 2>/tmp/err.txt
      if [ $? -eq 0 ]; then
        result="OK"
      else
        result="FAIL"
        err=`cat /tmp/err.txt`
      fi
      echo "$sample_type;$sample_exec;$result;$err"
    fi
  done
done

