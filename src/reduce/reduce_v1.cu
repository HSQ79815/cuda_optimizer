#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define DivUp(x, y) (x + y - 1) / y

template <typename T, size_t THREADS_PER_BLOCK> __global__ void Reduce(const T* src, T* dest, size_t N)
{
    __shared__ T sdata[THREADS_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int idx = tid + blockDim.x * blockIdx.x;

    sdata[tid] = idx < N ? src[idx] : T{};
    __syncthreads();

    for (unsigned int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dest[blockIdx.x] = sdata[0];
    }
}

template <typename T> bool Check(const T* s1, const T* s2, size_t N, float epsilon)
{
    for (size_t i = 0; i < N; ++i) {
        if (abs(s1[i] - s2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

template <typename T, size_t THREADS_PER_BLOCK> void ReduceCpu(const T* src, T* dest, size_t N)
{
    size_t num = N / THREADS_PER_BLOCK;
    size_t i   = 0;
    for (; i < num; ++i) {
        T      t        = T{};
        size_t base_idx = i * THREADS_PER_BLOCK;
        for (size_t j = 0; j < THREADS_PER_BLOCK; ++j) {
            t += src[base_idx + j];
        }
        dest[i] = t;
    }
    if (N % THREADS_PER_BLOCK != 0) {
        T& t = dest[i];
        t    = T{};
        for (size_t j = num * THREADS_PER_BLOCK; j < N; ++j) {
            t += src[j];
        }
    }
}

int main(int argc, char** argv)
{
    using type         = float;
    constexpr size_t N = 128 * 1024 * 1024;

    constexpr size_t THREADS_PER_BLOCK = 256;

    constexpr size_t  dest_N = DivUp(N, THREADS_PER_BLOCK);
    std::vector<type> cpu_src(N, 1.02);
    std::vector<type> cpu_dest(dest_N);

    ReduceCpu<type, THREADS_PER_BLOCK>(&cpu_src[0], &cpu_dest[0], N);

    type* device_src  = nullptr;
    type* device_dest = nullptr;

    std::vector<type> device_to_cpu(dest_N);

    cudaMalloc(&device_src, N * sizeof(type));
    cudaMalloc(&device_dest, dest_N * sizeof(type));

    cudaMemcpy(device_src, &cpu_src[0], N * sizeof(type), cudaMemcpyHostToDevice);

    dim3 block(THREADS_PER_BLOCK, 1, 1);
    dim3 grid(DivUp(N, THREADS_PER_BLOCK), 1, 1);
    Reduce<type, THREADS_PER_BLOCK><<<grid, block>>>(device_src, device_dest, N);

    float       elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    Reduce<type, THREADS_PER_BLOCK><<<grid, block>>>(device_src, device_dest, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaMemcpy(&device_to_cpu[0], device_dest, dest_N * sizeof(type), cudaMemcpyDeviceToHost);

    float epsilon = 1e-5;
    if (!Check(&cpu_dest[0], &device_to_cpu[0], dest_N, epsilon)) {
        std::cout << "result wrong!\n";
        std::cout << cpu_dest[0] << "\t" << cpu_dest[1] << "\t" << cpu_dest[2] << "\t" << cpu_dest[3] << "\t"
                  << cpu_dest[4] << "\t" << cpu_dest[5] << "\n";
        std::cout << device_to_cpu[0] << "\t" << device_to_cpu[1] << "\t" << device_to_cpu[2] << "\t"
                  << device_to_cpu[3] << "\t" << device_to_cpu[4] << "\t" << device_to_cpu[5] << "\n";
    }

    std::cout << "average cost: " << elapsedTime << " ms\n";
    float bandwidth = (N + dest_N) * sizeof(type) / elapsedTime * 1000 / (1 << 30);
    std::cout << "bandwidth: " << bandwidth << " GB/s\n";

    cudaFree(device_src);
    cudaFree(device_dest);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
