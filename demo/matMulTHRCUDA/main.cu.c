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
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */


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
    float *a, //a[NRA][NCA],           /* matrix A to be multiplied */
          *b, //b[NCA][NCB],           /* matrix B to be multiplied */
          *c; //c[NRA][NCB];           /* result matrix C */

    args arguments[2];
    pthread_t tid[2];

    numworkers = 2;
    
    srand((unsigned) time(0));

    a = (float*) malloc(NRA * NCA * sizeof (float));
    b = (float*) malloc(NCB * NCA * sizeof (float));
    c = (float*) malloc(NRA * NCB * sizeof (float));
    memset(c, 0, NRA * NCB * sizeof (float));
    memset(b, 0, NCB * NCA * sizeof (float));

    printf("a: %p b: %p c: %p\n", a, b, c);
    
    /**************************** master task ************************************/


    printf("Initializing a...\n");
    for (i = 0; i < NRA; i++)
        for (j = 0; j < NCA; j++)
            a[i * NCA + j] = rand()/RAND_MAX;
    printf("Initializing b...\n");
    for (i = 0; i < NCA; i++)
            b[i * NCB + i] = 1;

    /* Send matrix data to the worker tasks */
    averow = NRA / numworkers;
    extra = NRA % numworkers;
    offset = 0;
    for (int dest = 0; dest < numworkers; dest++) {
        arguments[dest].a = a + (offset * NCA);
        arguments[dest].b = b;
        arguments[dest].c = c + (offset * NCB);
    
        rows = averow;
        arguments[dest].dimsA.x = rows; //5*2*block_size;
        arguments[dest].dimsA.y = NCA;
        arguments[dest].dimsA.z = 1;
        arguments[dest].block_size = 32;

        arguments[dest].dimsB.x = NCA; //5*4*block_size;
        arguments[dest].dimsB.y = NCB; //5*2*block_size;
        arguments[dest].dimsB.z = 1;
        arguments[dest].devID = dest;
        
        offset += rows;
        
//        thread_entry((void *) &arguments[dest]);
        pthread_create(tid + dest, NULL, thread_entry,
               (void *) &arguments[dest]);

    }

    for (int dest = 0; dest < numworkers; dest++) {
       pthread_join(tid[dest], NULL);
       printf("Thread: %d joined\n", tid[dest]);
    }
    
    printf("Checking computed result for correctness: ");
    bool correct = true;

 

    for (int i = 0; i < (int)(NRA * NCB); i++)
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
//    printf("thread entered\n");
    args* argument = (args*) arg;
    matrixMultiply(argument->a, argument->b, argument->c, argument->block_size,
            argument->dimsA, argument->dimsB, argument->devID);    
}
