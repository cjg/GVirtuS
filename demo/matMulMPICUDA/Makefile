# Environment 
MKDIR=mkdir
CP=cp
CCADMIN=CCadmin
CC=/usr/bin/mpic++
NVCC=/usr/local/cuda/bin/nvcc
CFLAGS=-g -I/usr/local/cuda/include -I../common/inc
LIBS=-lcudart -lpthread
INCLUDE_DIR=-I/usr/local/cuda/include -I../common/inc
LIB_DIR=-L/usr/local/cuda/lib64
OBJ=main.cu.o matrixmul.o


all: matrixmul.o main.cu.o matMulMPI

matrixmul.o: matrixmul.cu
	$(NVCC) -c $(CFLAGS) $< -o $@ -ccbin $(CC) $(LIBS) $(INCLUDE_DIR) 

main.cu.o: main.cu.c
	$(CC) -c $(CFLAGS) $< -o $@ $(LIBS) $(INCLUDE_DIR) 

matMulMPI: matrixmul.o main.cu.o
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS) $(INCLUDE_DIR) $(LIB_DIR)

clean: 
	rm -f *.o matMulMPI

