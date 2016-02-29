/* 
 * File:   main.cu.c
 * Author: cpalmieri
 *
 * Created on November 11, 2015, 4:46 PM
 */

#include <mpi/mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define NRA 640                 /* number of rows in matrix A */
#define NCA 640                 /* number of columns in matrix A */
#define NCB 640                  /* number of columns in matrix B */
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

extern "C" {
    void matrixMultiply(float* a, float* b, float* c, int block_size, dim3 dimsA, dim3 dimsB);
}

int main(int argc, char *argv[]) {
    int numtasks, /* number of tasks in partition */
            taskid, /* a task identifier */
            numworkers, /* number of worker tasks */
            source, /* task id of message source */
            dest, /* task id of message destination */
            mtype, /* message type */
            rows, /* rows of matrix A sent to each worker */
            averow, extra, offset, /* used to determine rows sent to each worker */
            i, j, k, rc; /* misc */
    float *a, //a[NRA][NCA],           /* matrix A to be multiplied */
            *b, //b[NCA][NCB],           /* matrix B to be multiplied */
            *c; //c[NRA][NCB];           /* result matrix C */
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    numworkers = numtasks - 1;

    a = (float*) malloc(NRA * NCA * sizeof (float));
    b = (float*) malloc(NCB * NCA * sizeof (float));
    c = (float*) malloc(NRA * NCB * sizeof (float));

    //printf("%d: a: %x b: %x c: %x\n", taskid, a, b, c);

    /**************************** master task ************************************/
    if (taskid == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", numtasks);
        printf("Initializing arrays...\n");
        for (i = 0; i < NRA; i++)
            for (j = 0; j < NCA; j++)
                a[i * NCA + j] = rand() / RAND_MAX;
        for (i = 0; i < NCA; i++)
            b[i * NCB + i] = 1;

        /* Send matrix data to the worker tasks */
        averow = NRA / numworkers;
        extra = NRA % numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        for (dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
#ifdef DEBUG
            printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);
#endif
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(a + (offset * NCA), rows * NCA, MPI_FLOAT, dest, mtype,
                    MPI_COMM_WORLD);
            MPI_Send(b, NCA * NCB, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        /* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (i = 1; i <= numworkers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(c + (offset * NCB), rows * NCB, MPI_FLOAT, source, mtype,
                    MPI_COMM_WORLD, &status);
#ifdef DEBUG
            printf("Received results from task %d\n", source);
#endif
        }
        
        bool correct = true;

        for (int i = 0; i < (int) (NRA * NCB); i++) {
            if (a[i] != c[i]) {
                correct = false;
            }
        }

        printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    }

    /**************************** worker task ************************************/
    if (taskid > MASTER) {
        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
#ifdef DEBUG
        printf("%d: received offset: %d\n", taskid, offset);
#endif
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
#ifdef DEBUG
        printf("%d: received rows: %d\n", taskid, rows);
#endif
        MPI_Recv(a, rows * NCA, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
#ifdef DEBUG
        printf("%d: received a\n", taskid);
#endif
        MPI_Recv(b, NCA * NCB, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
#ifdef DEBUG
        printf("%d: b received\n", taskid);
#endif
        
        int block_size = 32;

        dim3 dimsA;
        dimsA.x = rows; //5*2*block_size;
        dimsA.y = NCA;
        dimsA.z = 1;
        dim3 dimsB;
        dimsB.x = NCA; //5*4*block_size;
        dimsB.y = NCB; //5*2*block_size;
        dimsB.z = 1;

        matrixMultiply(a, b, c, block_size, dimsA, dimsB);


        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(c, rows * NCB, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
}
