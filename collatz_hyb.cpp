#include <cstdio>
#include <algorithm>
#include <sys/time.h>
#include <mpi.h>


void GPU_Init();
void GPU_Exec(const long start, const long stop);
int GPU_Fini();

static int collatz(const long start, const long stop)
{
  int maxlen = 0;

  // todo: OpenMP code with default(none), a reduction, and a cyclic schedule (assume start to be odd) based on Project 4
  
  #pragma omp parallel for default(none) reduction(max: maxlen) schedule(static,1)      
  for(long i = start; i <= stop; i+=2)
  { 
     
  long val = i;
  int len = 1;

  while(val != 1)
      {    
         len++;
         if((val % 2) == 0)
           { val /= 2;}   //Even 
         else
           { val = 3 * val + 1; }  //odd 

      }
    maxlen = std::max(maxlen, len);
 }

  return maxlen;
}

int main(int argc, char *argv[])
{

  printf("Collatz v1.2\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s upper_bound cpu_percentage\n", argv[0]); exit(-1);}
  const long upper = atol(argv[1]);
  if (upper < 5) {fprintf(stderr, "ERROR: upper_bound must be at least 5\n"); exit(-1);}
  if ((upper % 2) != 1) {fprintf(stderr, "ERROR: upper_bound must be an odd number\n"); exit(-1);}
  const int percentage = atof(argv[2]);
  if ((percentage < 0) || (percentage > 100)) {fprintf(stderr, "ERROR: cpu_percentage must be between 0 and 100\n"); exit(-1);}





  int commSize, rankId;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

  const long cpu_start = (rankId * upper / commSize) | 1;
  const long gpu_stop = ((rankId +1)* upper / commSize) | 1;

  const long myRange = gpu_stop - cpu_start + 1;

  const long cpu_stop = (cpu_start + myRange * percentage / 100) & ~1LL;
  const long gpu_start = cpu_stop | 1;

  printf("upper bound: %ld\n", upper);
  printf("CPU percentage: %d\n", percentage);

  GPU_Init();


  MPI_Barrier(MPI_COMM_WORLD);

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // execute timed code
  GPU_Exec(gpu_start, gpu_stop);
  const int cpu_maxlen = collatz(cpu_start, cpu_stop);
  const int gpu_maxlen = GPU_Fini();
  const int maxlen = std::max(cpu_maxlen, gpu_maxlen);

// MPI_Reduce(dataToSend, resultToBeRecieved, sendCount, sendType, operation, targetProcessId, Comm)


  int maxLength;
  MPI_Reduce(&maxlen, &maxLength,1,MPI_INT,MPI_MAX ,0 ,MPI_COMM_WORLD);

 gettimeofday(&end, NULL);
 const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;

  if(rankId == 0){ 

   printf("Num of mpi processes:  %d \n", commSize);            
   printf("compute time: %4f s\n", runtime);
   printf("longest sequence: %d elements \n", maxLength);



     }

  // end time
 // gettimeofday(&end, NULL);
 // const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
 

// printf("longest sequence: %d elements \n", maxlen);
 
  MPI_Finalize();

 // printf("longest sequence: %d elements \n", maxlen);

  return 0;
}

