/* 
 * File:   main.cu.c
 * Author: cpalmieri
 *
 * Created on November 11, 2015, 4:46 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define NRA 320                 /* number of rows in matrix A */
#define NCA 320                 /* number of columns in matrix A */
#define NCB 320                  /* number of columns in matrix B */

void* thread_entry(void* arg);

extern "C" {
    void matrixMultiply(float* a, float* b, float* c, int block_size, dim3 dimsA, dim3 dimsB, int device);
}

typedef struct __args {
    float* a;
    float* b;
    float* c;
    int block_size;
    dim3 dimsA;
    dim3 dimsB;
    int devID;
} args;

int main(int argc, char *argv[]) {
    int numworkers, /* number of worker tasks */
            rows, /* rows of matrix A sent to each worker */
            averow, extra, offset, /* used to determine rows sent to each worker */
            i, j, k, rc; /* misc */
    int nra, nca, ncb;
    float *a, //a[NRA][NCA],           /* matrix A to be multiplied */
          *b, //b[NCA][NCB],           /* matrix B to be multiplied */
          *c; //c[NRA][NCB];           /* result matrix C */
    int nDevices;

    args* arguments;
    pthread_t* tid;
    
    printf("You can choose the number of threads and the size of matrixes\n"
            "calling %s num_threads rows_A cols_A rows_B\n\n", argv[0]);
    
    cudaGetDeviceCount(&nDevices);
    
    if (argc != 5) {
        numworkers = 2;
        nra = NRA;                
        nca = NCA;                
        ncb = NCB;                
    } else {
        sscanf(argv[1], "%d", &numworkers);
        sscanf(argv[2], "%d", &nra);
        sscanf(argv[3], "%d", &nca);
        sscanf(argv[4], "%d", &ncb);
    }
        
    
    arguments = (args*)malloc(numworkers * sizeof(args));
    tid = (pthread_t*)malloc(numworkers * sizeof(pthread_t));
    
    srand((unsigned) time(0));

    a = (float*) malloc(nra * nca * sizeof (float));
    b = (float*) malloc(ncb * nca * sizeof (float));
    c = (float*) malloc(nra * ncb * sizeof (float));
    memset(c, 0, nra * ncb * sizeof (float));
    memset(b, 0, ncb * nca * sizeof (float));

#ifdef DEBUG
    printf("a: %p b: %p c: %p\n", a, b, c);
#endif
    
    /**************************** master task ************************************/


    printf("Initializing a...\n");
    for (i = 0; i < nra; i++)
        for (j = 0; j < nca; j++)
            a[i * nca + j] = rand()/RAND_MAX;
    printf("Initializing b...\n");
    for (i = 0; i < nca; i++)
            b[i * ncb + i] = 1;

    /* Send matrix data to the worker tasks */
    averow = nra / numworkers;
    extra = nra % numworkers;
    offset = 0;
    for (int dest = 0; dest < numworkers; dest++) {
        
        arguments[dest].a = a + (offset * nca);
        arguments[dest].b = b;
        arguments[dest].c = c + (offset * ncb);
    
        rows = averow;
        arguments[dest].dimsA.x = rows; //5*2*block_size;
        arguments[dest].dimsA.y = nca;
        arguments[dest].dimsA.z = 1;
        arguments[dest].block_size = 32;

        arguments[dest].dimsB.x = nca; //5*4*block_size;
        arguments[dest].dimsB.y = ncb; //5*2*block_size;
        arguments[dest].dimsB.z = 1;
        arguments[dest].devID = dest % nDevices;
        
        offset += rows;
        
        pthread_create(tid + dest, NULL, thread_entry,
               (void *) &arguments[dest]);

    }
    
    
    for (int dest = 0; dest < numworkers; dest++) {
       pthread_join(tid[dest], NULL);
#ifdef DEBUG
       printf("Thread: %d joined\n", tid[dest]);
#endif
    }
    
    printf("Checking computed result for correctness: ");
    bool correct = true;

 

    for (int i = 0; i < (int)(nra * ncb); i++)
    {
        if (a[i] != c[i])
        {            
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
    
    if (correct)
        exit(EXIT_SUCCESS);
    else
        exit(EXIT_FAILURE);
    
}

void* thread_entry(void* arg) {
#ifdef DEBUG
    printf("thread entered\n");
#endif
    args* argument = (args*) arg;
    matrixMultiply(argument->a, argument->b, argument->c, argument->block_size,
            argument->dimsA, argument->dimsB, argument->devID);    
}

