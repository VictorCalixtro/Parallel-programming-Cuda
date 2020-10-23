#include <cstdio>
#include <cmath>
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static __global__ void fractal(const int width, const int start_frame, const int gpu_frames, unsigned char* const pic)
{
  // todo: use the GPU to compute the requested frames (base the code on the previous project)
}

unsigned char* GPU_Init(const int gpu_frames, const int width)
{
  unsigned char* d_pic;
  if (cudaSuccess != cudaMalloc((void **)&d_pic, gpu_frames * width * width * sizeof(unsigned char))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  return d_pic;
}

void GPU_Exec(const int start_frame, const int gpu_frames, const int width, unsigned char* d_pic)
{
  // todo: launch the kernel with ThreadsPerBlock and the appropriate number of blocks (do not wait for the kernel to finish)

FractalKernel<<<((gpu_frames - start_frame) * width + ThreadsPerBlock-1/ ThreadsPerBlock,ThreadsPerBlock>>>(start_frame,gpu_frame,width, d_pic);


}

void GPU_Fini(const int gpu_frames, const int width, unsigned char* pic, unsigned char* d_pic)
{
  // todo: copy the result from the device to the host and free the device memory
}

