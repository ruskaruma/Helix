#pragma once

#include "helix/common/types.hpp"
#include <memory>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace helix {

// Simple CUDA utility functions
namespace cuda_simple {
    bool isAvailable();
    int getDeviceCount();
    void setDevice(int deviceId);
    size_t getFreeMemory();
    void synchronize();
    
    // Memory management
    void* allocate(size_t size);
    void free(void* ptr);
    void memcpyHtoD(void* dst, const void* src, size_t size);
    void memcpyDtoH(void* dst, const void* src, size_t size);
}

// Simple CUDA IndexFlat implementation
class CudaIndexFlatSimple {
public:
    explicit CudaIndexFlatSimple(dim_t dimension, MetricType metric = MetricType::L2);
    ~CudaIndexFlatSimple();

    void train(const float* vectors, idx_t numVectors);
    void add(const float* vectors, idx_t numVectors);
    SearchResults search(const float* query, idx_t k) const;
    
    idx_t ntotal() const { return ntotal_; }
    dim_t dimension() const { return dimension_; }
    MetricType metric() const { return metric_; }
    bool isTrained() const { return trained_; }

private:
    void allocateGpuMemory();
    void freeGpuMemory();
    void syncToGpu();

    dim_t dimension_;
    MetricType metric_;
    idx_t ntotal_;
    bool trained_;
    
    // Host data
    std::vector<float> vectors_;
    
    // GPU data
    float* d_vectors_;
    float* d_queries_;
    float* d_distances_;
    idx_t* d_indices_;
    
    size_t gpu_memory_allocated_;
};

}
