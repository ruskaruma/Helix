#pragma once

#include "helix/common/types.hpp"
#include "helix/index/index_flat.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>

namespace helix {

class CudaIndexFlat : public IndexFlat {
public:
    explicit CudaIndexFlat(const IndexConfig& config);
    ~CudaIndexFlat() override;

    // Override search methods for GPU acceleration
    SearchResults search(const float* query, idx_t k) const override;
    void searchBatch(const float* queries, idx_t numQueries, idx_t k, 
                     std::vector<SearchResults>& results) const override;

    // GPU-specific methods
    void setDevice(int deviceId);
    int getDevice() const;
    size_t getGpuMemoryUsage() const;
    
    // Async operations
    void searchAsync(const float* query, idx_t k, cudaStream_t stream, 
                     SearchResults* result) const;
    void searchBatchAsync(const float* queries, idx_t numQueries, idx_t k,
                         cudaStream_t stream, std::vector<SearchResults>* results) const;

private:
    // GPU memory management
    void allocateGpuMemory();
    void freeGpuMemory();
    void syncToGpu();
    void syncFromGpu();

    // CUDA kernels
    void launchDistanceKernel(const float* queries, idx_t numQueries, 
                              float* distances, idx_t k) const;
    void launchTopKSelection(float* distances, idx_t* indices, 
                             idx_t numQueries, idx_t k) const;

    // GPU data
    float* d_vectors_;           // GPU vectors
    float* d_queries_;           // GPU queries
    float* d_distances_;         // GPU distance matrix
    idx_t* d_indices_;          // GPU result indices
    
    // CUDA handles
    cublasHandle_t cublas_handle_;
    cudaStream_t stream_;
    
    // Memory management
    size_t gpu_memory_allocated_;
    int device_id_;
    bool gpu_data_valid_;
    
    // Async support
    mutable std::vector<SearchResults> async_results_;
};

// CUDA utility functions
namespace cuda_utils {
    bool isCudaAvailable();
    int getDeviceCount();
    void setDevice(int deviceId);
    size_t getFreeMemory();
    size_t getTotalMemory();
    void synchronize();
    
    // Memory management
    void* allocate(size_t size);
    void free(void* ptr);
    void memcpyHtoD(void* dst, const void* src, size_t size);
    void memcpyDtoH(void* dst, const void* src, size_t size);
    void memcpyAsync(void* dst, const void* src, size_t size, cudaStream_t stream);
}

}
