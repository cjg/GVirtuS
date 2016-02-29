#mpirun -x LD_LIBRARY_PATH=/usr/local/cuda/lib  -np 2 matMulMPI

for i in 2 3 4
do
echo "Running via CUDA $i process"
time mpirun -x LD_LIBRARY_PATH=/usr/local/cuda/lib  -np $i -hostfile machine matMulMPI

echo "Running via GVirtuS $i process"
time mpirun -x LD_LIBRARY_PATH=/home/ubuntu/opt/lib/frontend  -np $i -hostfile machine matMulMPI
done
