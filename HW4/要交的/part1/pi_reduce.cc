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
    MPI_Status status;
    
    long long int local_toss = tosses / world_size, local_count = 0;
    if (world_rank == world_size - 1) local_toss += tosses % world_size;

    double x, y;
    unsigned int seed = time(NULL) * world_rank;
    //

    // TODO: use MPI_Reduce    
    // add by myself    
    for (long long int i = 0; i < local_toss; ++i)
    {
        x = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 + (-1.0);
        y = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 + (-1.0);
        if (x * x + y * y <= 1) local_count++;
    }
    double double_count = local_count; // !!! don't know whether declare a double local_count or this way is faster
    MPI_Reduce(&double_count, &pi_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //

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
