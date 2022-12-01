#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

#define THRESHOLD 0

int world_rank, world_size, ori_n;
MPI_Status status; // useless

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int partial_n_num;
    
    if (world_rank > 0)
    {
        MPI_Recv(&ori_n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(n_ptr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(m_ptr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(l_ptr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        MPI_Datatype rowtype;
        MPI_Type_contiguous(*m_ptr, MPI_INT, &rowtype);
        MPI_Type_commit(&rowtype);

        *a_mat_ptr = (int*) malloc(sizeof(int) * partial_n_num * *m_ptr + 100); // partial A
        *b_mat_ptr = (int*) malloc(sizeof(int) * *l_ptr * *m_ptr + 100);
        MPI_Recv(*a_mat_ptr, partial_n_num, rowtype, 0, 0, MPI_COMM_WORLD, &status); // partial A
        MPI_Recv(*b_mat_ptr, *l_ptr, rowtype, 0, 0, MPI_COMM_WORLD, &status);

        MPI_Type_free(&rowtype);
    }
    else if (world_rank == 0)
    {
        int err; // useless
        err = scanf("%d %d %d", n_ptr, m_ptr, l_ptr); // there can't exist space in the end of format string

        int n = *n_ptr, m = *m_ptr, l = *l_ptr;
        *a_mat_ptr = (int*) malloc(sizeof(int) * n * m + 100); // !!! don't know why need to + 1 or it error with malloc(): invalid size (unsorted)
        *b_mat_ptr = (int*) malloc(sizeof(int) * l * m + 100); // let b be transport, because we want the memory access to be continuous, so b is l row, m col

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                err = scanf("%d", *a_mat_ptr + i * m + j);

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < l; ++j)
                err = scanf("%d", *b_mat_ptr + j * m + i);

        // send msg to worker
        MPI_Datatype rowtype;
        MPI_Type_contiguous(*m_ptr, MPI_INT, &rowtype);
        MPI_Type_commit(&rowtype);

        int partial_n_start, partial_n_end;
        ori_n = n;
        for (int i = 1; i < world_size; ++i)
        {
            partial_n_start = n * i / world_size; 
            partial_n_end = n * (i + 1) / world_size;
            partial_n_num = partial_n_end - partial_n_start;

            MPI_Send(&ori_n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&partial_n_num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(m_ptr, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(l_ptr, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(*a_mat_ptr + partial_n_start * m, partial_n_num, rowtype, i, 0, MPI_COMM_WORLD); // partial A
            MPI_Send(*b_mat_ptr, l, rowtype, i, 0, MPI_COMM_WORLD);
        }

        MPI_Type_free(&rowtype);

        if (!(ori_n * m * l < THRESHOLD || ori_n < world_size)) 
            *n_ptr = ori_n / world_size; // rank 0 deal with partial A too
    }
}

void partial_matrix_multiply(const int n, const int m, const int l,
                             const int *a_mat, const int *b_mat, int *c_mat)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < l; ++j)
        {
            *(c_mat + i * l + j) = 0;
            for (int k = 0; k < m; ++k)
                *(c_mat + i * l + j) += *(a_mat + i * m + k) * *(b_mat + j * m + k);
        }
}

void partial_matrix_print(const int n, const int l, int *c_mat)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < l; ++j)
            printf("%d ", *(c_mat + i * l + j));
        printf("\n");
    }
}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    printf("rank %d ori %d %d %d %d %lld %d %d\n", world_rank, ori_n, n, m, l, (long long)ori_n * (long long)m * (long long)l, (long long)ori_n * m * l < THRESHOLD, (long long)ori_n * (long long)m * (long long)l < THRESHOLD);
    if (ori_n * m * l < THRESHOLD || ori_n < world_size)
    {
        if (world_rank == 0)
        {
            int *c_mat = (int*) malloc(sizeof(int) * n * l + 1000);
            partial_matrix_multiply(n, m, l, a_mat, b_mat, c_mat);
            partial_matrix_print(n, l, c_mat);

            free(c_mat);
        }
    }
    else
    {
        int *c_mat = (int*) malloc(sizeof(int) * n * l + 1000);
        partial_matrix_multiply(n, m, l, a_mat, b_mat, c_mat);

        // rank0 print, rank1 print, rank2...
        char sync; // useless
        if (world_rank == 0)
        {
            partial_matrix_print(n, l, c_mat);
            if (world_size > 1)
                MPI_Send(&sync, 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        }
        else if (world_rank > 0)
        {
            MPI_Recv(&sync, 1, MPI_CHAR, world_rank - 1, 0, MPI_COMM_WORLD, &status);
            partial_matrix_print(n, l, c_mat);
            if (world_rank != world_size - 1)
                MPI_Send(&sync, 1, MPI_CHAR, world_rank + 1, 0, MPI_COMM_WORLD);
        }
        free(c_mat);
    }
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    if (world_rank == 0)
    {
        free(a_mat);
        free(b_mat);
    }
}