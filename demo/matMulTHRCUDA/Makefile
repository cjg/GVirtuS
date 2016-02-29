# Environment 
MKDIR=mkdir
CP=cp
CCADMIN=CCadmin
CC=/usr/bin/g++
NVCC=/usr/local/cuda/bin/nvcc
CFLAGS=-g -I/usr/local/cuda/include -I/home/cpalmieri/NVIDIA_CUDA-6.5_Samples/common/inc
LIBS=-lcudart -lpthread
INCLUDE_DIR=-I/usr/local/cuda/include -I/home/cpalmieri/NVIDIA_CUDA-6.5_Samples/common/inc
LIB_DIR=-L/usr/local/cuda/lib64
OBJ=main.cu.o matrixmul.o


all: matrixmul.o main.cu.o matMulThr

matrixmul.o:
	$(NVCC) -c $(CFLAGS) matrixmul.cu -o $@ -ccbin $(CC) $(LIBS) $(INCLUDE_DIR) 

main.cu.o:
	$(CC) -c $(CFLAGS) main.cu.c -o $@ $(LIBS) $(INCLUDE_DIR) 

matMulThr: matrixmul.o main.cu.o
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS) $(INCLUDE_DIR) $(LIB_DIR)

clean: 
	rm -f *.o matMulThr 

