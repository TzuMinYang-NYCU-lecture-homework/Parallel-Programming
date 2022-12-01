#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    // add by myself
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Status status[world_size];
    
    long long int local_toss = tosses / world_size, local_count = 0;
    if (world_rank == world_size - 1) local_toss += tosses % world_size;

    double x, y;
    unsigned int seed = time(NULL) * world_rank;
    //

    if (world_rank > 0)
    {
        // TODO: MPI workers
        // add by myself
        for (long long int i = 0; i < local_toss; ++i)
        {
            x = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 + (-1.0);
            y = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 + (-1.0);
            if (x * x + y * y <= 1) local_count++; 
        }
        MPI_Send(&local_count, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);  // !!! can use send and irecv
        //
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // add by myself
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size - 1];
        long long local_count_arr[world_size - 1];
        
        for (int i = 1; i < world_size; ++i) // !!! remember from 1
            MPI_Irecv(&local_count_arr[i - 1], 1, MPI_LONG_LONG, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &requests[i - 1]);

        // put something between Irecv and wait so we can get some benefit
        for (long long int i = 0; i < local_toss; ++i)
        {
            x = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 + (-1.0);
            y = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 + (-1.0);
            if (x * x + y * y <= 1) local_count++; 
        }
        pi_result = local_count;

        // int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[])
        MPI_Waitall(world_size - 1, requests, status);
        for (int i = 0; i < world_size - 1; ++i)
            pi_result += local_count_arr[i];
        //
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        // add by myself
        pi_result = 4.0 * pi_result / (double)tosses;
        //

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
