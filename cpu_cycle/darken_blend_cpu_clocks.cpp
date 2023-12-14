#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h> /* Definition of HW_* constants */
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <sys/ioctl.h>
#include <unistd.h>
#include <array>

#if HABANA_TEST_API
#include "tensor.h"
#endif
#include "omp.h"

#define EXECUTOR_ERROR -1
#define EXECUTOR_SUCCESS 0

#define M 128
#define N 64
typedef std::array<std::array<float, N>, M> my2darray_t;

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags) {
  int ret;
  ret = syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  return ret;
}

#if HABANA_TEST_API
void darken_blend(
        const float_2DTensor& base,
        const float_2DTensor& active,
        float_2DTensor& out)
{
    int coords[5] = {0};

    int maxRows = out.Size(0);
    int maxCols = out.Size(1);

    for (int row = 0; row < maxRows; row++) {
        for (int col = 0; col < maxCols; col++) {
            coords[0] = row; coords[1] = col;
            if (base.ElementAt(coords) > active.ElementAt(coords)) {
                out.SetElement(coords, active.ElementAt(coords));
            } else {
                out.SetElement(coords, base.ElementAt(coords));
            }
        }
    }
}
#endif

static void darken_blend_ref(
        const my2darray_t& base,
        const my2darray_t& active,
        my2darray_t& out)
{
    for (size_t row = 0; row < M; row++) {
        #pragma omp simd
        for (size_t col = 0; col < N; col++) {
            if (base[row][col] > active[row][col]) {
                out[row][col] = active[row][col];
            } else {
                out[row][col] = base[row][col];
            }
        }
    }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << argv[0] << " <num_times_to_run_assembly_code> " << std::endl;
    return 0;
  }

  int counter = atoi(argv[1]);

  struct perf_event_attr pe_ins, pe_cycles;
  uint64_t count_ins, count_cycles;
  int fd_ins, fd_cycles;
  memset(&pe_ins, 0, sizeof(pe_ins));
  memset(&pe_cycles, 0, sizeof(pe_cycles));
  pe_ins.type = PERF_TYPE_HARDWARE;
  pe_ins.size = sizeof(pe_ins);
  pe_ins.config = PERF_COUNT_HW_INSTRUCTIONS;
  pe_ins.disabled = 1;
  pe_ins.exclude_kernel = 1;
  pe_ins.exclude_hv = 1;
  pe_cycles.type = PERF_TYPE_HARDWARE;
  pe_cycles.size = sizeof(pe_cycles);
  pe_cycles.config = PERF_COUNT_HW_REF_CPU_CYCLES;
  pe_cycles.disabled = 1;
  pe_cycles.exclude_kernel = 1;
  pe_cycles.exclude_hv = 1;

  pe_ins.pinned = 0;
  pe_cycles.pinned = 0;

  pid_t child_pid = 0;

  const int dim0  = 128;
  const int dim1  = 64;
  unsigned int tensor_shape[] = {dim0, dim1};

#if HABANA_TEST_API
  float_2DTensor base(tensor_shape);
  base.InitRand(-10.0f, 10.0f);
  float_2DTensor active(tensor_shape);
  active.InitRand(-10.0f, 10.0f);
  float_2DTensor out(tensor_shape);
#else
  auto init_tensor = [&](my2darray_t& a, float rangemin, float rangemax, unsigned seed=0) {
    srand(seed);
    for (size_t i = 0; i < M; i++)
      for (size_t j = 0; j < N; j++)
        a[i][j] = (rangemin + (float) rand() / ((float) RAND_MAX / ((float) rangemax - rangemin + (float) 1) + (float) 1));
  };

  my2darray_t base;
  init_tensor(base, -10.0f, 10.0f, 0);
  my2darray_t active;
  init_tensor(active, -10.0f, 10.0f, 0);
  my2darray_t out;
#endif
  
  fd_ins = perf_event_open(&pe_ins, child_pid, -1, -1, 0);
  fd_cycles = perf_event_open(&pe_cycles, child_pid, -1, -1, 0);
  ioctl(fd_ins, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd_cycles, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd_ins, PERF_EVENT_IOC_ENABLE, 0);
  ioctl(fd_cycles, PERF_EVENT_IOC_ENABLE, 0);

  for (int i = 0; i < counter; i++) {
#ifdef HABANA_TEST_API
    darken_blend(base, active, out);
#else
    darken_blend_ref(base, active, out);
#endif
  }

  ioctl(fd_cycles, PERF_EVENT_IOC_DISABLE, 0);
  ioctl(fd_ins, PERF_EVENT_IOC_DISABLE, 0);

  read(fd_ins, &count_ins, sizeof(count_ins));
  read(fd_cycles, &count_cycles, sizeof(count_cycles));

  float cpi = (float) count_cycles / (float) count_ins;
  float ipc = (float) count_ins / (float) count_cycles;

  std::cout << "insn=" << argv[1] << ",count_insns=" << count_ins << ",count_cycles=" << count_cycles
            << ",cpi=" << std::setprecision(2) << cpi
            << ",ipc=" << std::setprecision(2) << ipc
            << std::endl;

  close(fd_ins);
  close(fd_cycles);

return 0;
}
