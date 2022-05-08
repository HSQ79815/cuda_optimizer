
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

#define DivUp(x, y) ((x + y - 1) / y)

template <typename T> void __global__ MatrixCopy2D(const T* src, int M, int N, T* dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        int idx  = y * N + x;
        dst[idx] = src[idx];
    }
}

template <typename T> void __global__ MatrixCopy1D(const T* src, int N, T* dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x < N)
        dst[x] = src[x];
}

template <typename T, int NUM> void __global__ MatrixCopy1DVec(const T* src, int N, T* dst)
{
    int x = blockIdx.x * blockDim.x * NUM + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < NUM; ++i,x += blockDim.x) {
        if(x < N)
            dst[x] = src[x];
    }
}

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();

    using type = float;

    constexpr int M  = 1 << 13;
    constexpr int N  = 1 << 13;
    constexpr int MN = M * N;

    constexpr int size_in_bytes = MN * sizeof(type);

    std::vector<type> cpu_src(MN, type(1));

    type* device_src  = nullptr;
    type* device_dest = nullptr;

    cudaMalloc(&device_src, size_in_bytes);
    cudaMalloc(&device_dest, size_in_bytes);

    cudaMemcpy(device_src, &cpu_src[0], size_in_bytes, cudaMemcpyHostToDevice);

    constexpr int thread_per_block_in_x = 32;
    constexpr int thread_per_block_in_y = 32;
    constexpr int thread_per_block      = thread_per_block_in_x * thread_per_block_in_y;

    float       elapsedTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int loop = 100;

    {
        dim3 block(thread_per_block_in_x, thread_per_block_in_y, 1);
        dim3 grid(DivUp(M, thread_per_block_in_x), DivUp(N, thread_per_block_in_y), 1);
        MatrixCopy2D<type><<<grid, block>>>(device_src, M, N, device_dest);

        cudaEventRecord(start, 0);
        for (int i = 0; i < loop; ++i)
            MatrixCopy2D<type><<<grid, block>>>(device_src, M, N, device_dest);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        float avg_cost = elapsedTime / loop;
        LOG(INFO) << "================2D: " << thread_per_block_in_x << " x " << thread_per_block_in_y
                  << "==================";
        LOG(INFO) << "average cost: " << avg_cost << " ms";
        float bandwidth = size_in_bytes * 2 / avg_cost * 1000 / (1 << 30);
        LOG(INFO) << "bandwidth: " << bandwidth << " GB/s";
    }

    {
        dim3 block(thread_per_block, 1, 1);
        dim3 grid(DivUp(MN, thread_per_block), 1, 1);
        MatrixCopy1D<type><<<grid, block>>>(device_src, MN, device_dest);

        cudaEventRecord(start, 0);

        for (int i = 0; i < loop; ++i)
            MatrixCopy1D<type><<<grid, block>>>(device_src, MN, device_dest);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        float avg_cost = elapsedTime / loop;
        LOG(INFO) << "================1D: " << thread_per_block << "==================";
        LOG(INFO) << "average cost: " << avg_cost << " ms";
        float bandwidth = size_in_bytes * 2 / avg_cost * 1000 / (1 << 30);
        LOG(INFO) << "bandwidth: " << bandwidth << " GB/s";
    }

    {
        constexpr int NUM = 2;
        dim3          block(thread_per_block, 1, 1);
        dim3          grid(DivUp(DivUp(MN, NUM), thread_per_block), 1, 1);
        MatrixCopy1DVec<type, NUM><<<grid, block>>>(device_src, MN, device_dest);

        cudaEventRecord(start, 0);

        for (int i = 0; i < loop; ++i)
            MatrixCopy1DVec<type, NUM><<<grid, block>>>(device_src, MN, device_dest);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        float avg_cost = elapsedTime / loop;
        LOG(INFO) << "================1D: " << thread_per_block << " NUM:" << NUM << "==================";
        LOG(INFO) << "average cost: " << avg_cost << " ms";
        float bandwidth = size_in_bytes * 2 / avg_cost * 1000 / (1 << 30);
        LOG(INFO) << "bandwidth: " << bandwidth << " GB/s";
    }

    {
        constexpr int NUM = 4;
        dim3          block(thread_per_block, 1, 1);
        dim3          grid(DivUp(DivUp(MN, NUM), thread_per_block), 1, 1);
        MatrixCopy1DVec<type, NUM><<<grid, block>>>(device_src, MN, device_dest);

        cudaEventRecord(start, 0);

        for (int i = 0; i < loop; ++i)
            MatrixCopy1DVec<type, NUM><<<grid, block>>>(device_src, MN, device_dest);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        float avg_cost = elapsedTime / loop;
        LOG(INFO) << "================1D: " << thread_per_block << " NUM:" << NUM << "==================";
        LOG(INFO) << "average cost: " << avg_cost << " ms";
        float bandwidth = size_in_bytes * 2 / avg_cost * 1000 / (1 << 30);
        LOG(INFO) << "bandwidth: " << bandwidth << " GB/s";
    }

    {
        constexpr int NUM = 8;
        dim3          block(thread_per_block, 1, 1);
        dim3          grid(DivUp(DivUp(MN, NUM), thread_per_block), 1, 1);
        MatrixCopy1DVec<type, NUM><<<grid, block>>>(device_src, MN, device_dest);

        cudaEventRecord(start, 0);

        for (int i = 0; i < loop; ++i)
            MatrixCopy1DVec<type, NUM><<<grid, block>>>(device_src, MN, device_dest);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        float avg_cost = elapsedTime / loop;
        LOG(INFO) << "================1D: " << thread_per_block << " NUM:" << NUM << "==================";
        LOG(INFO) << "average cost: " << avg_cost << " ms";
        float bandwidth = size_in_bytes * 2 / avg_cost * 1000 / (1 << 30);
        LOG(INFO) << "bandwidth: " << bandwidth << " GB/s";
    }
    cudaFree(device_src);
    cudaFree(device_dest);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
