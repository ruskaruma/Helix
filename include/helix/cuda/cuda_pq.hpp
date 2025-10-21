#pragma once

#include "helix/common/types.hpp"
#include "helix/index/index_pq.hpp"
#include "helix/quantization/product_quantizer.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>

namespace helix {

class CudaIndexPQ : public IndexPQ {
public:
    explicit CudaIndexPQ(const IndexConfig& config);
    ~CudaIndexPQ() override;

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

    // CUDA kernels for PQ operations
    void launchPQEncodeKernel(const float* queries, idx_t numQueries,
                             uint8_t* codes) const;
    void launchADCDistanceKernel(const uint8_t* query_codes, 
                                const uint8_t* database_codes,
                                float* distances, idx_t numQueries, idx_t k) const;
    void launchTopKSelection(float* distances, idx_t* indices, 
                             idx_t numQueries, idx_t k) const;

    // GPU data
    uint8_t* d_codes_;           // GPU PQ codes
    uint8_t* d_query_codes_;     // GPU query codes
    float* d_distances_;         // GPU distance matrix
    idx_t* d_indices_;          // GPU result indices
    
    // PQ-specific GPU data
    float* d_centroids_;         // GPU centroids
    float* d_distance_tables_;  // GPU distance tables
    
    // CUDA handles
    cublasHandle_t cublas_handle_;
    cudaStream_t stream_;
    
    // Memory management
    size_t gpu_memory_allocated_;
    int device_id_;
    bool gpu_data_valid_;
    
    // PQ-specific parameters
    idx_t code_size_;
    idx_t subvector_size_;
};

// CUDA PQ utility functions
namespace cuda_pq_utils {
    // Distance table computation
    void computeDistanceTables(const float* centroids, idx_t numCentroids,
                              idx_t subvectorSize, float* distanceTables);
    
    // ADC distance computation
    void computeADCDistances(const uint8_t* queryCodes, const uint8_t* databaseCodes,
                            const float* distanceTables, float* distances,
                            idx_t numQueries, idx_t numVectors, idx_t codeSize);
    
    // Memory management for PQ
    void allocatePQMemory(idx_t numVectors, idx_t codeSize, 
                         uint8_t** d_codes, float** d_distance_tables);
    void freePQMemory(uint8_t* d_codes, float* d_distance_tables);
}

}
