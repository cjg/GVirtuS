#mpirun -x LD_LIBRARY_PATH=/usr/local/cuda/lib  -np 2 matMulMPI


for j in 800 1600 3200
do
  for i in 2 3 4
  do
    echo "Running via CUDA $i process"
    time mpirun -x LD_LIBRARY_PATH=/usr/local/cuda/lib  -np $i -hostfile machine matMulMPI $j $j $j 

    echo "Running via GVirtuS $i process"
    time mpirun -x LD_LIBRARY_PATH=/home/ubuntu/opt/lib/frontend  -np $i -hostfile machine matMulMPI $j $j $j
  done
done
