#include "helix/cuda/cuda_flat.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

namespace helix {
namespace cuda_utils {

bool isCudaAvailable() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        return false;
    }
    return deviceCount > 0;
}

int getDeviceCount() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        return 0;
    }
    return deviceCount;
}

void setDevice(int deviceId) {
    cudaError_t error = cudaSetDevice(deviceId);
    if (error != cudaSuccess) {
        throw HelixException("Failed to set CUDA device: " + std::string(cudaGetErrorString(error)));
    }
}

size_t getFreeMemory() {
    size_t free, total;
    cudaError_t error = cudaMemGetInfo(&free, &total);
    if (error != cudaSuccess) {
        return 0;
    }
    return free;
}

size_t getTotalMemory() {
    size_t free, total;
    cudaError_t error = cudaMemGetInfo(&free, &total);
    if (error != cudaSuccess) {
        return 0;
    }
    return total;
}

void synchronize() {
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        throw HelixException("CUDA synchronization failed: " + std::string(cudaGetErrorString(error)));
    }
}

void* allocate(size_t size) {
    void* ptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memory allocation failed: " + std::string(cudaGetErrorString(error)));
    }
    return ptr;
}

void free(void* ptr) {
    if (ptr) {
        cudaError_t error = cudaFree(ptr);
        if (error != cudaSuccess) {
            throw HelixException("CUDA memory free failed: " + std::string(cudaGetErrorString(error)));
        }
    }
}

void memcpyHtoD(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memcpy HtoD failed: " + std::string(cudaGetErrorString(error)));
    }
}

void memcpyDtoH(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memcpy DtoH failed: " + std::string(cudaGetErrorString(error)));
    }
}

void memcpyAsync(void* dst, const void* src, size_t size, cudaStream_t stream) {
    cudaError_t error = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memcpy async failed: " + std::string(cudaGetErrorString(error)));
    }
}

// Additional utility functions
void checkCudaError(const char* file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string msg = "CUDA error at " + std::string(file) + ":" + std::to_string(line) + 
                         " - " + std::string(cudaGetErrorString(error));
        throw HelixException(msg);
    }
}

void printDeviceInfo() {
    int deviceCount = getDeviceCount();
    std::cout << "CUDA Device Count: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    }
}

bool isDeviceCompatible(int deviceId) {
    if (deviceId < 0 || deviceId >= getDeviceCount()) {
        return false;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    
    // Check compute capability (require 7.5+ for optimal performance)
    return prop.major > 7 || (prop.major == 7 && prop.minor >= 5);
}

void optimizeDevice(int deviceId) {
    if (deviceId < 0 || deviceId >= getDeviceCount()) {
        throw HelixException("Invalid device ID for optimization");
    }
    
    cudaSetDevice(deviceId);
    
    // Set device flags for optimal performance
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
}

}
}
