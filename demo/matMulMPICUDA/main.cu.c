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

//#define DEBUG

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
    int nra, nca, ncb;
    float *a, //a[NRA][NCA],           /* matrix A to be multiplied */
            *b, //b[NCA][NCB],           /* matrix B to be multiplied */
            *c; //c[NRA][NCB];           /* result matrix C */
    MPI_Status status;
    


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (taskid == 0)
      printf("You can choose the size of matrixes calling %s rows_A cols_A rows_B\n\n", argv[0]);

    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }


    if (argc == 4) {
        sscanf(argv[1], "%d", &nra);
        sscanf(argv[2], "%d", &nca);
        sscanf(argv[3], "%d", &ncb);
    } else {
        nra = NRA;
        nca = NCA;
        ncb = NCB;
    }

#ifdef DEBUG
    printf("arg: %d, nra: %d,  nca: %d, ncb: %d\n", argc, nra, nca, ncb);
#endif 
    numworkers = numtasks - 1;

    a = (float*) malloc(nra * nca * sizeof (float));
    b = (float*) malloc(ncb * nca * sizeof (float));
    c = (float*) malloc(nra * ncb * sizeof (float));

    /**************************** master task ************************************/
    if (taskid == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", numtasks);
        printf("Initializing arrays...\n");
        for (i = 0; i < nra; i++)
            for (j = 0; j < nca; j++)
                a[i * nca + j] = rand() / RAND_MAX;
        for (i = 0; i < nca; i++)
            b[i * ncb + i] = 1;

        /* Send matrix data to the worker tasks */
        averow = nra / numworkers;
        extra = nra % numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        for (dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
#ifdef DEBUG
            printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);
#endif
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(a + (offset * nca), rows * nca, MPI_FLOAT, dest, mtype,
                    MPI_COMM_WORLD);
            MPI_Send(b, nca * ncb, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        /* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (i = 1; i <= numworkers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(c + (offset * ncb), rows * ncb, MPI_FLOAT, source, mtype,
                    MPI_COMM_WORLD, &status);
#ifdef DEBUG
            printf("Received results from task %d\n", source);
#endif
        }
        
        bool correct = true;

        for (int i = 0; i < (int) (nra * ncb); i++) {
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
        MPI_Recv(a, rows * nca, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
#ifdef DEBUG
        printf("%d: received a\n", taskid);
#endif
        MPI_Recv(b, nca * ncb, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
#ifdef DEBUG
        printf("%d: b received\n", taskid);
#endif
        
        int block_size = 32;

        dim3 dimsA;
        dimsA.x = rows; 
        dimsA.y = nca;
        dimsA.z = 1;
        dim3 dimsB;
        dimsB.x = nca; 
        dimsB.y = ncb; 
        dimsB.z = 1;

        matrixMultiply(a, b, c, block_size, dimsA, dimsB);


        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(c, rows * ncb, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
}

