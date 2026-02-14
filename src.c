#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
#include<math.h>
#include<float.h>

int main(int argc, char *argv[])
{
    int TAG_D1 = 1;
    int TAG_D2 = 2;
    int TAG_D1_BACK = 3;
    int TAG_D2_BACK = 4;
    int TAG_MAXD = 5;
    
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (argc != 6) 
    {
        if (rank == 0) 
        {
            fprintf(stderr, "Usage: %s M D1 D2 T seed\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // verify whether the arguments we pass exist.
    int M = atoi(argv[1]);
    int D1 = atoi(argv[2]);
    int D2 = atoi(argv[3]);
    int T = atoi(argv[4]);
    int seed = atoi(argv[5]);

    if (M <= 0 || D1 <= 0 || D2 <= D1 || T < 0)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Invalid inputs. Need: M>0, D1>0, D2>D1, T>=0.\n");
        }
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
    double *data_received_D1 = (double*)malloc(M * sizeof(double));
    double *data_received_D2 = (double*)malloc(M * sizeof(double));
    double *data_at_D1 = (double*)malloc(M * sizeof(double)); 
    double *data_at_D2 = (double*)malloc(M * sizeof(double)); 

    double *buffer_updated_for_D1 = (double*)malloc(M * sizeof(double));
    double *buffer_updated_for_D2 = (double*)malloc(M * sizeof(double));

    srand(seed); 
    for (int i=0; i<M; i++)
    {
        data_received_D1[i] = (double)rand()*(rank+1)/10000.0;
        data_received_D2[i] = data_received_D1[i];
    }
    
    double sTime;
    MPI_Barrier(MPI_COMM_WORLD);
    sTime = MPI_Wtime();

    for (int t=0; t<T; t++)
    {

        if (l1 >= 0)// Receiver rank receives and sends back
        {
            MPI_Recv(data_received, M, MPI_DOUBLE, l1, TAG_D1, MPI_COMM_WORLD, &status); // receive from sender at l1

            for (int i=0; i<M; i++)
                data_at_D1[i] = data_received[i] * data_received[i];

            MPI_Send(data_at_D1, M, MPI_DOUBLE, l1, TAG_D1_BACK, MPI_COMM_WORLD); // send back to sender at l1
            
        } 
        
        if (l2 >= 0)// Receiver rank receives and sends back
        {
            MPI_Recv(data_received, M, MPI_DOUBLE, l2, TAG_D2, MPI_COMM_WORLD, &status); // receive from sender at l2

            for (int i=0; i<M; i++)
                data_at_D2[i] = log(data_received[i]);

            MPI_Send(data_at_D2, M, MPI_DOUBLE, l2, TAG_D2_BACK, MPI_COMM_WORLD); // send back to sender at l2
        }


        if (r1 < P)// Sender rank sends
        {
            MPI_Send(data_received_D1, M, MPI_DOUBLE, r1, TAG_D1, MPI_COMM_WORLD); // send to receiver at r1
            MPI_Recv(data_received, M, MPI_DOUBLE, r1, TAG_D1_BACK, MPI_COMM_WORLD, &status); // receive from receiver at r1
            
            if(t < T-1)
            {
                for (int i = 0; i<M; i++) // compute buffer to update data_received before next iteration
                    buffer_updated_for_D1[i] = (unsigned long long) data_received[i] % 100000;
            }
            else
            {
                for (int i = 0; i<M; i++)
                    buffer_updated_for_D1[i] = data_received[i];
            }

            for(int i = 0; i<M; i++) // update data_received using buffer
                data_received_D1[i] = buffer_updated_for_D1[i];
        }
        
        if (r2 < P)// Sender rank sends
        {
            MPI_Send(data_received_D2, M, MPI_DOUBLE, r2, TAG_D2, MPI_COMM_WORLD); // send to receiver at r2
            MPI_Recv(data_received, M, MPI_DOUBLE, r2, TAG_D2_BACK, MPI_COMM_WORLD, &status); // receive from receiver at r2

            if(t < T-1)
            {
                for (int i = 0; i<M; i++) // compute buffer to update data_received before next iteration
                    buffer_updated_for_D2[i] = data_received[i] * 100000.0;
            }
            else
            {
                for (int i = 0; i<M; i++)
                    buffer_updated_for_D2[i] = data_received[i];
            }
            
            
            for(int i = 0; i<M; i++) // update data_received using buffer
                data_received_D2[i] = buffer_updated_for_D2[i];
        }
        
    }
    
    // Calculate maximum of data_received_D1 and data_received_D2 for all processes
    double local_max_D1 = data_received_D1[0], local_max_D2 = data_received_D2[0];
    for(int i = 0; i < M; i++)
    {
        if(data_received_D1[i] > local_max_D1) local_max_D1 = data_received_D1[i];
        if(data_received_D2[i] > local_max_D2) local_max_D2 = data_received_D2[i];
    }
    

    double global_max_D1 = -DBL_MAX, global_max_D2 = -DBL_MAX;
    if(r1 < P || r2 < P) 
    {
        if (rank == 0) 
        {
          global_max_D1 = local_max_D1;
          global_max_D2 = local_max_D2;
        } 
        else // only valid senders send the max D1/D2 data to rank 0 
        {
          double maxD[2] = { local_max_D1, local_max_D2 };
          MPI_Send(maxD, 2, MPI_DOUBLE, 0, TAG_MAXD, MPI_COMM_WORLD);
        }
    }
    
    if (rank == 0)// calculates global maximum D1/D2 for all the max D1/D2 received from valid senders
    {
        int num_valid_senders = P - D1; // since D1 is smaller than D2, any rank that sends to D1 is to be considered valid
        if (num_valid_senders < 0) num_valid_senders = 0;
        
        for(int i = 1; i < num_valid_senders; i++)
        {
            double maxD_it[2];
            MPI_Recv(maxD_it, 2, MPI_DOUBLE, i, TAG_MAXD, MPI_COMM_WORLD, &status);

            if(maxD_it[0] > global_max_D1) global_max_D1 = maxD_it[0];
            if(maxD_it[1] > global_max_D2) global_max_D2 = maxD_it[1];
        }
    }
    
    
    

    double eTime = MPI_Wtime();
    double local_time = eTime - sTime;

    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    
    if (rank == 0)
    {
            printf("P = %d, M = %d, D1 = %d, D2 = %d, T = %d, seed = %d\n",P,M,D1,D2,T,seed);                                                                                             
            printf("Max D1: %.15e, Max D2: %.15e, Time: %.6f\n\n", global_max_D1, global_max_D2, max_time);
    }

    free(data_received);
    free(data_received_D1);
    free(data_received_D2);
    free(data_at_D1);
    free(data_at_D2);
    free(buffer_updated_for_D1);
    free(buffer_updated_for_D2);

    MPI_Finalize();

    return 0;
}