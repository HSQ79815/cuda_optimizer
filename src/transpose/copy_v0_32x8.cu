
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define DivUp(x, y) (x + y - 1) / y

template <typename T> void __global__ MatrixCopy0(const T* src, int M, int N, T* dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        int idx = y * N + x;
        dst[idx] = src[idx];
    }
}

int main (int argc, char** argv){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();

    using type         = float;

    constexpr int M = 1 << 13;
    constexpr int N = 1 << 13;

    constexpr int size_in_bytes = M * N * sizeof(type);

    std::vector<type> cpu_src(M*N, type(1));

    type* device_src  = nullptr;
    type* device_dest = nullptr;

    cudaMalloc(&device_src, size_in_bytes);
    cudaMalloc(&device_dest, size_in_bytes);

    cudaMemcpy(device_src, &cpu_src[0],size_in_bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 8, 1);
    dim3 grid(DivUp(M, 32), DivUp(N, 8), 1);
    MatrixCopy0<<<grid,block>>>(device_src,M,N,device_dst);

    float       elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    MatrixCopy0<<<grid,block>>>(device_src,M,N,device_dst);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaMemcpy(&device_to_cpu[0], device_dest, dest_N * sizeof(type), cudaMemcpyDeviceToHost);

    LOG(INFO) << "average cost: " << elapsedTime << " ms";
    float bandwidth = size_in_bytes * 2 / elapsedTime * 1000 / (1 << 30);
    LOG(INFO) << "bandwidth: " << bandwidth << " GB/s";

    cudaFree(device_src);
    cudaFree(device_dest);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
