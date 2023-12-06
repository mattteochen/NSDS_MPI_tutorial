#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Set DEBUG 1 if you want car movement to be deterministic
#define DEBUG 1

FILE *fp;

const int num_segments = 256;

const int num_iterations = 1000;
const int count_every = 10;

const double alpha = 0.5;
const int max_in_per_sec = 10;

// Returns the number of car that enter the first segment at a given iteration.
int create_random_input() {
#if DEBUG
  return 1;
#elif
  return rand() % max_in_per_sec;
#endif
}

// Returns 1 if a car needs to move to the next segment at a given iteration, 0 otherwise.
int move_next_segment() {
#if DEBUG
  return 1;
#elif
  return rand() < alpha ? 1 : 0;
#endif
}

void move_cars(const int rank, const int num_procs, const int size, int* road, MPI_Request* req) {
  for (int i=size-1; i>=0; i--) {
    int current_cars = road[i];
    int car_to_send = 0;
    for (int car=0; car<current_cars; car++) {
      car_to_send += move_next_segment();
    }

    fprintf(fp, "rank %d sends %d cars from %d to %d\n", rank, car_to_send, i, i+1);

    if (i == size-1) {
      const int next_process = rank+1;
      if (next_process < num_procs) {
        MPI_Isend(&car_to_send, 1, MPI_INT, next_process, 0, MPI_COMM_WORLD, req);
      }
    } else {
      road[i+1] += car_to_send;
    }
    road[i] -= car_to_send;
  }
}

void receive_cars(const int rank, const int num_procs, const int size, int* road) {
  int received_cars = 0;
  MPI_Recv(&received_cars, 1, MPI_INT, rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  road[0] += received_cars;
}

int main(int argc, char** argv) { 
  MPI_Init(NULL, NULL);

  int rank;
  int num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  srand(time(NULL) + rank);
  
  // Produce the output file
  char file_name_buff[100] = {0};
  strcat(&file_name_buff[0], "./out_");
  char file_name_num_buff[2];
  file_name_num_buff[0] = rank + '0';
  file_name_num_buff[1] = '\0';
  strcat(&file_name_buff[4], (const char*)&file_name_num_buff[0]);
  printf("rank %d file name %s\n", rank, file_name_buff);

  fp = fopen(file_name_buff, "w");
  
  // TODO: define and init variables
  const int km_for_process = num_segments / num_procs;
  int* road = malloc(sizeof(int) * km_for_process);
  memset(road, 0, sizeof(int) * km_for_process);

  // Simulate for num_iterations iterations
  for (int it = 0; it < num_iterations; ++it) {
    MPI_Request req;
    fprintf(fp, "iteration n: %d\n", it);

    // Move cars across segments
    move_cars(rank, num_procs, km_for_process, road, &req);

    if (rank == 0) {
      int new_cars = create_random_input();
      fprintf(fp, "new cars to enter n: %d\n", new_cars);
      road[0] += new_cars;
    }

    if (rank) {
      receive_cars(rank, num_procs, km_for_process, road);
    }
    if (rank < num_procs-1) {
      MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    // When needed, compute the overall sum
    if (it%count_every == 0) {
      int global_sum = 0;
      int local_sum = 0;
      for (int i=0; i<km_for_process; i++) local_sum += road[i];
      fprintf(fp, "rank %d local_sum %d\n", rank, local_sum);
      MPI_Reduce(&local_sum, &global_sum, num_procs, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
      
      if (rank == 0) {
        fprintf(fp, "Iteration: %d, sum: %d\n", it, global_sum);
      }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // TODO deallocate dynamic variables, if needed
  free(road);
  fclose(fp); //Don't forget to close the file when finished
  
  MPI_Finalize();
}
