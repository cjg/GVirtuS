#! /bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage $0 NVIDIA_SAMPLES_PATH output_file [avoid_file]"
    exit -1
fi

base_dir=$1
output_file=$2
avoid_file=""
if [ "$#" -eq 3 ]; then
  avoid_file=$3
  if [ ! -f $avoid_file ]; then
    echo "Avoid file doesn't exist"
    exit -1
  fi
  _avoid_file="/tmp/avoid-$RANDOM-.txt"
  cat $avoid_file | sed '/^#/ d' >$_avoid_file
fi

_output_file="/tmp/output-$RANDOM-.txt"
echo >$_output_file "TYPE;SAMPLE;RESULT;TIME;ERROR"

cd $base_dir
sample_type_paths=`ls -d ${base_dir}/?_*`
for sample_type_path in $sample_type_paths
do
  sample_paths=`ls -d ${sample_type_path}/*`
  sample_type=`basename $sample_type_path`
  echo "In $sample_type_path ..."
  for sample_path in $sample_paths
  do
    sample_exec=`basename $sample_path`
    cd $sample_path
    if [ -f "$sample_exec" ]; then
      avoid=""
      if [ -n "$avoid_file" ]; then
        avoid=`grep -w $sample_type/$sample_exec $_avoid_file`
      fi
      if [ -z "$avoid" ]; then
        _stderr_file="/tmp/stderr-$RANDOM-.txt"
        echo "Testing: $sample_type/$sample_exec"
        err=""
        result="FAIL"
        dT=-1
        t0=`echo $(($(date +%s%N)/1000000))`
        ./$sample_exec >/dev/null 2>$_stderr_file
        if [ $? -ne 0 ]; then
          err_lines=`cat $_stderr_file | wc | awk '{ print $1 }'`
          skip_err_lines=`grep "no version information available" $_stderr_file | wc | awk '{ print $1 }'`
          if [ "$err_lines" -gt "$skip_err_lines" ]; then
            err=`tr '\n' ';' <$_stderr_file`
            echo "FAIL:"
            cat $_stderr_file
          fi
        else
          result="OK"
          t1=`echo $(($(date +%s%N)/1000000))`
          dT=$((t1-t0))
        fi
        rm $_stderr_file 2>/dev/null
        echo "----------------------------"
        echo >>$_output_file "$sample_type;$sample_exec;$result;$dT;$err"
      fi
    fi
  done
done
cp $_output_file $output_file
rm $_output_file 2>/dev/null
rm $_avoid_file 2>/dev/null
