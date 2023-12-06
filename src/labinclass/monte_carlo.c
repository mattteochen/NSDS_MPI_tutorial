#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

const int num_iter_per_proc = 10 * 1000 * 1000;

int main() {
  MPI_Init(NULL, NULL);
    
  int rank;
  int num_procs;
  int sum = 0;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  srand(time(NULL) + rank);

  int* counts_buffer = 0;
  if (rank == 0) {
    counts_buffer = malloc(sizeof(int) * num_procs);
    memset(counts_buffer, 0, sizeof(int) * num_procs);
  }

  int local_count = 0;
  for (unsigned i=0; i<num_iter_per_proc; ++i) {
    double x = (1.0 / RAND_MAX) * rand();
    double y = (1.0 / RAND_MAX) * rand();
    // printf("x: %.3f y: %.3f\n", x, y);
    if (x*x + y*y <= 1) local_count++;
  }

  // MPI_Gather(&local_count, 1, MPI_INT, counts_buffer, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_count, &counts_buffer[0], num_procs, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // TODO
  if (rank == 0) {
    // If using gather:
    // for (unsigned i=0; i<num_procs; i++) {
    //   sum += counts_buffer[i];
    // }
    // else:
    sum = counts_buffer[0];
    double pi = (4.0*sum) / (num_iter_per_proc*num_procs);
    printf("Pi = %f\n", pi);
    free(counts_buffer);
  }
    
  MPI_Finalize();
  return 0;
}
