#include <cstdio>
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static int* d_maxlen;

static __global__ void collatz(const long start, const long stop, int* const maxlen)
{
  // todo: process odd values from start (assume start to be odd) to stop (inclusively if stop is odd) with one thread per value (based on code from previous project)
 

 

 const long i = threadIdx.x + blockIdx.x * (long)blockDim.x;


 if(i+start  < stop  ) // Each thread does work if and only if less than the stop value
  {
    long val = 2*(i +((start-1)/2))+1;
    int len = 1;
    while(val != 1){
        len++;
        if((val % 2) == 0)//even
          {val = val / 2;} 
        else //Odd
          {val = 3 * val +1;}

} if(len > *maxlen){ atomicMax(maxlen, len);} // If greater than greatest length, becomes new max len;

  }

}

void GPU_Init()
{
  int maxlen = 0;
  if (cudaSuccess != cudaMalloc((void **)&d_maxlen, sizeof(int))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMemcpy(d_maxlen, &maxlen, sizeof(int), cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
}

void GPU_Exec(const long start, const long stop)
{
  if (start <= stop) {
    collatz<<<((stop - start + 2) / 2 + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(start, stop, d_maxlen);
  }
}

int GPU_Fini()
{
  int maxlen;

  // todo: copy the result from the device to the host and free the device memory
  
  

  if(cudaSuccess != cudaMemcpy(&maxlen, d_maxlen, sizeof(int), cudaMemcpyDeviceToHost)){fprintf(stderr, "Error: copying to host failed\n"); exit(-1);}


  cudaFree(d_maxlen);
  return maxlen;
}

