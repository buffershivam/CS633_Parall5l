#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
#include<math.h>
// ! Possible Domain Error in log()
// ! ask if sentinel approach is better
// ! do we have to sync up all the processes at the end of each iteration? or is that handled by the logic. i cant tell.
// ! edge cases test: D1 = 0, P = 1, M = 0 etc. 
int main(int argc, char *argv[]) {
    int TAG_D1 = 1;
    int TAG_D2 = 2;
    int TAG_M21 = 3;
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc < 6) {
        printf("Usage: %s M D1 D2 T seed\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    // verify whether the arguments we pass exist.
    int M = atoi(argv[1]);
    int D1 = atoi(argv[2]);
    int D2 = atoi(argv[3]);
    int T = atoi(argv[4]);
    int seed = atoi(argv[5]);

    if (D1 >= D2) {
        printf("Usage: D1 < D2 \n");
        MPI_Finalize();
        return 1;
    }

    int P;
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int l1 = rank - D1;
    int r1 = rank + D1;

    int l2 = rank - D2;
    int r2 = rank + D2;

    MPI_Status status;
    
    double *data_received = (double*)malloc(M * sizeof(double)); // sender to D1 and D2
    double *data_at_D1 = (double*)malloc(M * sizeof(double)); // receiver from D1 
    double *data_at_D2 = (double*)malloc(M * sizeof(double)); // receiver from D2

    double *l1_recv_buffer = (double*)malloc(M * sizeof(double));
    double *l2_recv_buffer = (double*)malloc(M * sizeof(double));
    // ! do array elements arrive out of order?

    srand(seed); 
    for (int i=0; i<M; i++)
        data_received[i] = (double)rand()/1000;
    
    double sTime;
    MPI_Barrier(MPI_COMM_WORLD);
    sTime = MPI_Wtime();
    for (int j=0; j<T; j++) {

        if (r1 < P) {
            MPI_Send(data_received, M, MPI_DOUBLE, r1, TAG_D1, MPI_COMM_WORLD);
            // printf("Rank %d -> Rank %d (D1 right send)\n", rank, r1);
            MPI_Recv(data_at_D1, M, MPI_DOUBLE, r1, TAG_D1, MPI_COMM_WORLD, &status);
        }
        if (l1 >= 0) {
            MPI_Recv(l1_recv_buffer, M, MPI_DOUBLE, l1, TAG_D1, MPI_COMM_WORLD, &status);
            for (int i=0; i<M; i++) {
                l1_recv_buffer[i] *= l1_recv_buffer[i]; // squared
            }
            MPI_Send(l1_recv_buffer, M, MPI_DOUBLE, l1, TAG_D1, MPI_COMM_WORLD);
            // printf("Rank %d -> Rank %d (D1 left send)\n", rank, l1);
        } // check if alternating speeds it up

        if (r2 < P) {
            MPI_Send(data_received, M, MPI_DOUBLE, r2, TAG_D2, MPI_COMM_WORLD);
            // printf("Rank %d -> Rank %d (D2 right send)\n", rank, r2);
            MPI_Recv(data_at_D2, M, MPI_DOUBLE, r2, TAG_D2, MPI_COMM_WORLD, &status);
        }
        if (l2 >= 0) {
            MPI_Recv(l2_recv_buffer, M, MPI_DOUBLE, l2, TAG_D2, MPI_COMM_WORLD, &status);
            for (int i=0; i<M; i++) {
                l2_recv_buffer[i] = log(l2_recv_buffer[i]); // log
            }
            MPI_Send(l2_recv_buffer, M, MPI_DOUBLE, l2, TAG_D2, MPI_COMM_WORLD);
            // printf("Rank %d -> Rank %d (D2 left send)\n", rank, l2);
        }

        // compare data_received with data_at_D1 and data_at_D2 to verify correctness
        // printf("Iter %d:\n", j);
        // printf("Rank %d original data_received:\n", rank);
        // for (int i=0; i<M; i++)         printf(" %f", data_received[i]);
        // printf("\n");
        
        // printf("Rank %d received data_at_D1:\n", rank);
        // for (int i=0; i<M; i++)        printf(" %f", data_at_D1[i]);
        // printf("\n");

        // printf("Rank %d received data_at_D2:\n", rank);
        // for (int i=0; i<M; i++)        printf(" %f", data_at_D2[i]);
        // printf("\n");
        
        if (r1 < P) {
            for (int i = 0; i < M; i++) {
                data_received[i] += data_at_D1[i];
            }
            if (r2 < P) {
                for (int i = 0; i < M; i++) {
                    data_received[i] += data_at_D2[i];
                }
            }
        }
        // printf("Rank %d modified data_received:\n", rank);
        // for (int i=0; i<M; i++)         printf(" %f", data_received[i]);
        // printf("\n\n");
    }
    
    // many to one send
    int num_valid_senders = P - D1; 
    double *final_data = malloc(num_valid_senders * M * sizeof(double));
    double *recv_buffer = malloc(M * sizeof(double));
    if (rank == 0) {
        for (int i = 0; i < M; i++) {
            final_data[i] = data_received[i];
        }
        for (int i = 1; i < num_valid_senders; i++) {
            MPI_Recv(recv_buffer, M, MPI_DOUBLE, i, TAG_M21, MPI_COMM_WORLD, &status);
            for (int j = 0; j < M; j++) {
                final_data[i * M + j] = recv_buffer[j];
            }
        }
    } 
    else if (rank < num_valid_senders) {
        MPI_Send(data_received, M, MPI_DOUBLE, 0, TAG_M21, MPI_COMM_WORLD);
    }
    
    double sum = 0.0;
    double max = final_data[0];

    for (int i = 0; i < num_valid_senders * M; i++) {
        sum += final_data[i];

        if (final_data[i] > max)
            max = final_data[i];
    }

    double avg = sum / (num_valid_senders * M);
    
    free(final_data);
    free(recv_buffer);
    free(data_received);
    free(data_at_D1);
    free(data_at_D2);
    free(l1_recv_buffer);
    free(l2_recv_buffer);

    MPI_Barrier(MPI_COMM_WORLD);
    double eTime = MPI_Wtime();
    MPI_Finalize();
    if (rank == 0)     printf("Max: %f, Avg: %f, Time: %f\n", max, avg, eTime - sTime);

    return 0;
}