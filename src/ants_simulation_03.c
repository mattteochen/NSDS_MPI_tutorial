#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Group number: 03
 *
 * Group members
 *  - Carlo Ronconi
 *  - Giulia Prosio
 *  - Kaixi Matteo Chen
 */

// Notes: we have opted for double precision floating point types as we have
// seen small deviations in the corvegence point when using different number of
// processes while using 32 bit floats

#define DEBUG 0

const double min = 0;
const double max = 1000;
const double len = max - min;
const int num_ants = 8 * 1000 * 1000;
const int num_food_sources = 10;
const int num_iterations = 500;

const double d2_multiplier = 0.012;
const double d1_multiplier = 0.01;

double random_position() {
  return (double)rand() / (double)(RAND_MAX / (max - min)) + min;
}

/*
 * Process 0 invokes this function to initialize food sources.
 */
void init_food_sources(double *food_sources) {
  for (int i = 0; i < num_food_sources; i++) {
    food_sources[i] = random_position();
  }
}

/*
 * Process 0 invokes this function to initialize the position of ants.
 */
void init_ants(double *ants) {
  for (int i = 0; i < num_ants; i++) {
    ants[i] = random_position();
  }
}

// allocate and populate gloabl ants
void initilise_gloabal_ants(double **buff) {
  *buff = malloc(sizeof(double) * num_ants);
  if (!*buff) {
    printf("malloc error for ants\n");
    exit(1);
  }
  init_ants(*buff);
}

double compute_buffer_mean(const int count, const double *buff) {
  double sum = 0.0;
  for (int i = 0; i < count; i++) {
    sum += buff[i];
  }
  return sum / (double)count;
}

double compute_min_food_source(const double ant_position,
                               const double *food_sources) {
  double min_distance = DBL_MAX;
  for (int i = 0; i < num_food_sources; ++i) {
    double diff = food_sources[i] - ant_position;
    if (fabs(diff) < fabs(min_distance)) {
      min_distance = diff;
    }
  }
  return min_distance;
}

void local_update(const int managed_ants, double *ants, const double center,
                  const double *food_sources) {
  for (int i = 0; i < managed_ants; ++i) {
    double *ant = &ants[i];
    double f1 = d1_multiplier * compute_min_food_source(*ant, food_sources);
    double f2 = d2_multiplier * (center - *ant);
    *ant += f1 + f2;
  }
}

int main() {
  MPI_Init(NULL, NULL);

  int rank;
  int num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  srand(rank);

  // only rank 0 allocates and populates the global ants buffer
  // every process allocate the food souces buffer but only process zero
  // populate it
  double *ants = 0;
  double *food_sources = malloc(sizeof(double) * num_food_sources);
  ;
  if (!rank) {
    initilise_gloabal_ants(&ants);
#if DEBUG == 1
    printf("initial global ants:\n");
    for (int i = 0; i < num_ants; ++i) {
      printf("%.3f ", ants[i]);
    }
    printf("\n");
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
  if (!rank) {
    init_food_sources(food_sources);
  }
  MPI_Bcast(food_sources, num_food_sources, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#if DEBUG == 0
  for (int i = 0; i < num_procs; i++) {
    if (rank == i) {
      printf("rank: %u below my food sources:\n", rank);
      for (int j = 0; j < num_food_sources; ++j) {
        printf("%.3f ", food_sources[j]);
      }
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  // every process allocated the space for its managed ants
  const int num_ants_per_process = num_ants / num_procs;
  double *local_ants = malloc(sizeof(double) * num_ants_per_process);
  if (!local_ants) {
    printf("malloc failed for local_ants\n");
    exit(1);
  }
  // ants distribuition
  MPI_Scatter(ants, num_ants_per_process, MPI_DOUBLE, local_ants,
              num_ants_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#if DEBUG == 1
  for (int i = 0; i < num_procs; i++) {
    if (rank == i) {
      printf("rank: %u\n", rank);
      for (int j = 0; j < num_ants_per_process; ++j) {
        printf("%.3f ", local_ants[j]);
      }
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  // compute the fist center
  double center = 0;
  if (!rank) {
    center = compute_buffer_mean(num_ants, ants);
  }
  MPI_Bcast(&center, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#if DEBUG == 1
  printf("rank %u inital centr %.3f\n", rank, center);
#endif

  // Iterative simulation
  for (int iter = 0; iter < num_iterations; iter++) {
    local_update(num_ants_per_process, local_ants, center, food_sources);
    double local_center = compute_buffer_mean(num_ants_per_process, local_ants);
#if DEBUG == 1
    printf("rank %u local center: %.3f\n", rank, local_center);
#endif

    MPI_Reduce(&local_center, &center, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
      center /= (double)num_procs;
      printf("Iteration: %d - Average position: %f\n", iter, center);
    }
    MPI_Bcast(&center, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // Free memory
  free(local_ants);
  free(food_sources);
  if (!rank) {
    free(ants);
  }

  MPI_Finalize();
  return 0;
}
