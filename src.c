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
    int TAG_MD1 = 3;
    int TAG_MD2 = 4;
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc < 6)
    {
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

    if (D1 >= D2)
    {
        printf("Usage: D1 < D2 \n");
        MPI_Finalize();
        return 1;
    }

    int P;
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int l1 = rank - D1;
    int l2 = rank - D2;
    
    int r1 = rank + D1;
    int r2 = rank + D2;

    MPI_Status status;
    
    double *data_received = (double*)malloc(M * sizeof(double));
    double *data_at_D1 = (double*)malloc(M * sizeof(double)); 
    double *data_at_D2 = (double*)malloc(M * sizeof(double)); 

    double *buffer_updated_for_D1 = (double*)malloc(M * sizeof(double));
    double *buffer_updated_for_D2 = (double*)malloc(M * sizeof(double));

    srand(seed); 
    for (int i=0; i<M; i++)
        data_received[i] = (double)rand()*(rank+1)/10000.0;
    
    double sTime;
    MPI_Barrier(MPI_COMM_WORLD);
    sTime = MPI_Wtime();

    for (int j=0; j<T; j++)
    {

        if (r1 < P)// Sender rank
        {
            MPI_Send(buffer_updated_for_D1, M, MPI_DOUBLE, r1, TAG_D1, MPI_COMM_WORLD);

            for (int i = 0; i<M; i++)
                buffer_updated_for_D1[i] = (unsigned long long) data_received[i] % 100000;

            MPI_Recv(data_received, M, MPI_DOUBLE, r1, TAG_D1, MPI_COMM_WORLD, &status);
        }
        if (r2 < P)// Sender rank
        {
            MPI_Send(buffer_updated_for_D2, M, MPI_DOUBLE, r2, TAG_D2, MPI_COMM_WORLD);

            for (int i = 0; i<M; i++)
                buffer_updated_for_D2[i] = data_received[i] * 100000;

            MPI_Recv(data_received, M, MPI_DOUBLE, r2, TAG_D2, MPI_COMM_WORLD, &status);
        }
        if (l1 >= 0)// Receiver rank
        {
            MPI_Recv(data_received, M, MPI_DOUBLE, l1, TAG_D1, MPI_COMM_WORLD, &status);

            for (int i=0; i<M; i++)
                data_at_D1[i] = data_received[i] * data_received[i];

            MPI_Send(data_at_D1, M, MPI_DOUBLE, l1, TAG_D1, MPI_COMM_WORLD);
            
        } 
        
        if (l2 >= 0)// Receiver rank
        {
            MPI_Recv(data_received, M, MPI_DOUBLE, l2, TAG_D2, MPI_COMM_WORLD, &status);

            for (int i=0; i<M; i++)
                data_at_D2[i] = log(data_received[i]);

            MPI_Send(data_at_D2, M, MPI_DOUBLE, l2, TAG_D2, MPI_COMM_WORLD);
        }

        
    }
    
    // Calculate maximum of data_at_D1 and data_at_D2 for all processes
    double max_D1 = data_at_D1[0], max_D2 = data_at_D2[0];
    double global_max_D1 = -1e18, global_max_D2 = -1e18;
    for(int i = 0; i < M; i++)
    {
        if(data_at_D1[i] > max_D1) max_D1 = data_at_D1[i];
        if(data_at_D2[i] > max_D2) max_D2 = data_at_D2[i];
    }
    
    if (rank == 0)
    {
        for(int i = 1; i < P; i++)
        {
            double max_D1_it,max_D2_it;
            MPI_Recv(&max_D1_it, 1, MPI_DOUBLE, i, TAG_MD1, MPI_COMM_WORLD, &status);
            MPI_Recv(&max_D2_it, 1, MPI_DOUBLE, i, TAG_MD2, MPI_COMM_WORLD, &status);

            if(max_D1_it > global_max_D1) global_max_D1 = max_D1_it;
            if(max_D2_it > global_max_D2) global_max_D2 = max_D2_it;
        }
    }
    else
    {
        MPI_Send(&max_D1, 1, MPI_DOUBLE, 0, TAG_MD1, MPI_COMM_WORLD);
        MPI_Send(&max_D2, 1, MPI_DOUBLE, 0, TAG_MD2, MPI_COMM_WORLD);
    }
    
    
    free(data_received);
    free(data_at_D1);
    free(data_at_D2);
    free(buffer_updated_for_D1);
    free(buffer_updated_for_D2);

    MPI_Barrier(MPI_COMM_WORLD);
    double eTime = MPI_Wtime();
    MPI_Finalize();
    if (rank == 0)     printf("Max D1: %f, Max D2: %f, Time: %f\n", global_max_D1, global_max_D2, eTime - sTime);

    return 0;
}