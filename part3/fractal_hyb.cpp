#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <sys/time.h>
#include "BMP43805351.h"

unsigned char* GPU_Init(const int gpu_frames, const int width);
void GPU_Exec(const int start_frame, const int gpu_frames, const int width, unsigned char* d_pic);
void GPU_Fini(const int gpu_frames, const int width, unsigned char* pic, unsigned char* d_pic);

static void fractal(const int start_frame, const int cpu_frames, const int width, unsigned char* const pic)
{
  // todo: use OpenMP to parallelize the for-row loop with default(none) and do not specify a schedule
}

int main(int argc, char *argv[])
{
  // set up MPI
  int comm_sz, my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) printf("Fractal v1.9\n");

  // check command line
  if (argc != 4) {fprintf(stderr, "USAGE: %s frame_width cpu_frames gpu_frames\n", argv[0]); exit(-1);}
  const int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "ERROR: frame_width must be at least 10\n"); exit(-1);}
  const int cpu_frames = atoi(argv[2]) / comm_sz;
  if (cpu_frames < 0) {fprintf(stderr, "ERROR: cpu_frames must be at least 0\n"); exit(-1);}
  const int gpu_frames = atoi(argv[3]) / comm_sz;
  if (gpu_frames < 0) {fprintf(stderr, "ERROR: gpu_frames must be at least 0\n"); exit(-1);}
  const int frames = cpu_frames + gpu_frames;
  if (frames < 1) {fprintf(stderr, "ERROR: total number of frames must be at least 1\n"); exit(-1);}

  const int cpu_start_frame = my_rank * frames;
  const int gpu_start_frame = cpu_start_frame + cpu_frames;

  if (my_rank == 0) {
    printf("cpu_frames: %d\n", cpu_frames * comm_sz);
    printf("gpu_frames: %d\n", gpu_frames * comm_sz);
    printf("frames: %d\n", frames * comm_sz);
    printf("width: %d\n", width);
    printf("MPI tasks: %d\n", comm_sz);
  }

  // allocate picture arrays
  unsigned char* pic = new unsigned char [frames * width * width];
  unsigned char* d_pic = GPU_Init(gpu_frames, width);
  unsigned char* full_pic = NULL;
  if (my_rank == 0) full_pic = new unsigned char [frames * comm_sz * width * width];

  // start time
  timeval start, end;
  MPI_Barrier(MPI_COMM_WORLD);  // for better timing
  gettimeofday(&start, NULL);

  // asynchronously compute the requested frames on the GPU
  GPU_Exec(gpu_start_frame, gpu_frames, width, d_pic);

  // compute the remaining frames on the CPU
  fractal(cpu_start_frame, cpu_frames, width, pic);

  // copy the GPU's result into the appropriate location of the CPU's pic array
  GPU_Fini(gpu_frames, width, &pic[cpu_frames * width * width], d_pic);

  // gather the resulting frames

  // todo: gather the results into full_pic on compute node 0

  if (my_rank == 0) {
    gettimeofday(&end, NULL);
    const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("compute time: %.4f s\n", runtime);

    // write result to BMP files
    if ((width <= 257) && (frames * comm_sz <= 60)) {
      for (int frame = 0; frame < frames * comm_sz; frame++) {
        BMP24 bmp(0, 0, width - 1, width - 1);
        for (int y = 0; y < width - 1; y++) {
          for (int x = 0; x < width - 1; x++) {
            const int p = full_pic[frame * width * width + y * width + x];
            const int e = full_pic[frame * width * width + y * width + (x + 1)];
            const int s = full_pic[frame * width * width + (y + 1) * width + x];
            const int dx = std::min(2 * std::abs(e - p), 255);
            const int dy = std::min(2 * std::abs(s - p), 255);
            bmp.dot(x, y, dx * 0x000100 + dy * 0x000001);
          }
        }
        char name[32];
        sprintf(name, "fractal%d.bmp", frame + 1000);
        bmp.save(name);
      }
    }

    delete [] full_pic;
  }

  MPI_Finalize();
  delete [] pic;
  return 0;
}

