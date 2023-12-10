#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

const int num_iter_per_proc = 10 * 1000 * 1000;

int main() {
  MPI_Init(NULL, NULL);
    
  int rank;
  int num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  srand(time(NULL) + rank);
    
  double x, y;
  int count = 0;
  for (int i=0; i<num_iter_per_proc; i++) {
    x = ((double) rand()) / RAND_MAX;
    y = ((double) rand()) / RAND_MAX;
    if ((x*x) + (y*y) <= 1.0) {
      count++;
    }
  }

  int sum;
  MPI_Reduce(&count, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
  if (rank == 0) {
    double pi = (4.0*sum) / (num_iter_per_proc*num_procs);
    printf("Pi = %f\n", pi);
  }
    
  MPI_Finalize();
  return 0;
}
