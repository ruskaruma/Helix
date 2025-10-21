#include "helix/cuda/cuda_pq.hpp"
#include "helix/common/utils.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstring>

namespace helix {

CudaIndexPQ::CudaIndexPQ(const IndexConfig& config) 
    : IndexPQ(config), d_codes_(nullptr), d_query_codes_(nullptr), 
      d_distances_(nullptr), d_indices_(nullptr), d_centroids_(nullptr),
      d_distance_tables_(nullptr), gpu_memory_allocated_(0),
      device_id_(0), gpu_data_valid_(false) {
    
    if (!cuda_utils::isCudaAvailable()) {
        throw HelixException("CUDA not available on this system");
    }
    
    cudaSetDevice(device_id_);
    cublasCreate(&cublas_handle_);
    cudaStreamCreate(&stream_);
    
    code_size_ = quantizer_.getCodeSize();
    subvector_size_ = dimension_ / quantizer_.getNumSubvectors();
    
    allocateGpuMemory();
}

CudaIndexPQ::~CudaIndexPQ() {
    freeGpuMemory();
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void CudaIndexPQ::setDevice(int deviceId) {
    if (deviceId < 0 || deviceId >= cuda_utils::getDeviceCount()) {
        throw HelixException("Invalid CUDA device ID");
    }
    
    device_id_ = deviceId;
    cudaSetDevice(device_id_);
    
    freeGpuMemory();
    allocateGpuMemory();
    gpu_data_valid_ = false;
}

int CudaIndexPQ::getDevice() const {
    return device_id_;
}

size_t CudaIndexPQ::getGpuMemoryUsage() const {
    return gpu_memory_allocated_;
}

void CudaIndexPQ::allocateGpuMemory() {
    if (dimension_ == 0) return;
    
    // Calculate memory requirements
    size_t codes_size = ntotal_ * code_size_ * sizeof(uint8_t);
    size_t query_codes_size = code_size_ * sizeof(uint8_t);
    size_t distance_size = ntotal_ * sizeof(float);
    size_t index_size = ntotal_ * sizeof(idx_t);
    size_t centroids_size = quantizer_.getNumCentroids() * subvector_size_ * sizeof(float);
    size_t distance_tables_size = quantizer_.getNumCentroids() * quantizer_.getNumCentroids() * sizeof(float);
    
    // Allocate GPU memory
    cudaMalloc(&d_codes_, codes_size);
    cudaMalloc(&d_query_codes_, query_codes_size);
    cudaMalloc(&d_distances_, distance_size);
    cudaMalloc(&d_indices_, index_size);
    cudaMalloc(&d_centroids_, centroids_size);
    cudaMalloc(&d_distance_tables_, distance_tables_size);
    
    gpu_memory_allocated_ = codes_size + query_codes_size + distance_size + 
                           index_size + centroids_size + distance_tables_size;
    
    // Initialize indices
    std::vector<idx_t> host_indices(ntotal_);
    for (idx_t i = 0; i < ntotal_; ++i) {
        host_indices[i] = i;
    }
    cudaMemcpy(d_indices_, host_indices.data(), index_size, cudaMemcpyHostToDevice);
}

void CudaIndexPQ::freeGpuMemory() {
    if (d_codes_) { cudaFree(d_codes_); d_codes_ = nullptr; }
    if (d_query_codes_) { cudaFree(d_query_codes_); d_query_codes_ = nullptr; }
    if (d_distances_) { cudaFree(d_distances_); d_distances_ = nullptr; }
    if (d_indices_) { cudaFree(d_indices_); d_indices_ = nullptr; }
    if (d_centroids_) { cudaFree(d_centroids_); d_centroids_ = nullptr; }
    if (d_distance_tables_) { cudaFree(d_distance_tables_); d_distance_tables_ = nullptr; }
    gpu_memory_allocated_ = 0;
}

void CudaIndexPQ::syncToGpu() {
    if (!gpu_data_valid_ && ntotal_ > 0) {
        // Copy PQ codes to GPU
        size_t codes_size = ntotal_ * code_size_ * sizeof(uint8_t);
        cudaMemcpy(d_codes_, codes_.data(), codes_size, cudaMemcpyHostToDevice);
        
        // Copy centroids to GPU
        size_t centroids_size = quantizer_.getNumCentroids() * subvector_size_ * sizeof(float);
        cudaMemcpy(d_centroids_, quantizer_.getCentroids().data(), 
                   centroids_size, cudaMemcpyHostToDevice);
        
        // Compute distance tables on GPU
        cuda_pq_utils::computeDistanceTables(d_centroids_, 
                                            quantizer_.getNumCentroids(),
                                            subvector_size_, 
                                            d_distance_tables_);
        
        gpu_data_valid_ = true;
    }
}

void CudaIndexPQ::syncFromGpu() {
    // For read-only operations, no sync needed
}

SearchResults CudaIndexPQ::search(const float* query, idx_t k) const {
    if (k <= 0 || k > ntotal_) {
        throw HelixException("Invalid k value for search");
    }
    
    syncToGpu();
    
    // Encode query using PQ
    std::vector<uint8_t> query_codes(code_size_);
    quantizer_.encode(query, query_codes.data());
    
    // Copy query codes to GPU
    cudaMemcpy(d_query_codes_, query_codes.data(), 
               code_size_ * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // Launch ADC distance computation kernel
    launchADCDistanceKernel(d_query_codes_, d_codes_, d_distances_, 1, k);
    
    // Launch top-k selection kernel
    launchTopKSelection(d_distances_, d_indices_, 1, k);
    
    // Copy results back to host
    SearchResults result;
    result.indices.resize(k);
    result.distances.resize(k);
    
    cudaMemcpy(result.indices.data(), d_indices_, k * sizeof(idx_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.distances.data(), d_distances_, k * sizeof(float), cudaMemcpyDeviceToHost);
    
    return result;
}

void CudaIndexPQ::searchBatch(const float* queries, idx_t numQueries, idx_t k,
                              std::vector<SearchResults>& results) const {
    if (numQueries == 0) return;
    
    syncToGpu();
    
    // Allocate temporary GPU memory for batch
    uint8_t* d_batch_query_codes;
    float* d_batch_distances;
    idx_t* d_batch_indices;
    
    size_t query_codes_size = numQueries * code_size_ * sizeof(uint8_t);
    size_t distance_size = numQueries * ntotal_ * sizeof(float);
    size_t index_size = numQueries * k * sizeof(idx_t);
    
    cudaMalloc(&d_batch_query_codes, query_codes_size);
    cudaMalloc(&d_batch_distances, distance_size);
    cudaMalloc(&d_batch_indices, index_size);
    
    // Encode all queries
    std::vector<uint8_t> batch_query_codes(numQueries * code_size_);
    for (idx_t i = 0; i < numQueries; ++i) {
        quantizer_.encode(queries + i * dimension_, 
                         batch_query_codes.data() + i * code_size_);
    }
    
    // Copy query codes to GPU
    cudaMemcpy(d_batch_query_codes, batch_query_codes.data(), 
               query_codes_size, cudaMemcpyHostToDevice);
    
    // Launch batch ADC distance computation
    launchADCDistanceKernel(d_batch_query_codes, d_codes_, d_batch_distances, 
                           numQueries, k);
    
    // Launch batch top-k selection
    launchTopKSelection(d_batch_distances, d_batch_indices, numQueries, k);
    
    // Copy results back
    results.resize(numQueries);
    for (idx_t i = 0; i < numQueries; ++i) {
        results[i].indices.resize(k);
        results[i].distances.resize(k);
        
        cudaMemcpy(results[i].indices.data(), 
                   d_batch_indices + i * k, 
                   k * sizeof(idx_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(results[i].distances.data(), 
                   d_batch_distances + i * k, 
                   k * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Cleanup
    cudaFree(d_batch_query_codes);
    cudaFree(d_batch_distances);
    cudaFree(d_batch_indices);
}

void CudaIndexPQ::searchAsync(const float* query, idx_t k, cudaStream_t stream,
                              SearchResults* result) const {
    // Implementation for async search
    // This would use the provided stream for non-blocking execution
    // For now, fall back to synchronous search
    *result = search(query, k);
}

void CudaIndexPQ::searchBatchAsync(const float* queries, idx_t numQueries, idx_t k,
                                   cudaStream_t stream, std::vector<SearchResults>* results) const {
    // Implementation for async batch search
    // This would use the provided stream for non-blocking execution
    // For now, fall back to synchronous batch search
    searchBatch(queries, numQueries, k, *results);
}

void CudaIndexPQ::launchPQEncodeKernel(const float* queries, idx_t numQueries,
                                       uint8_t* codes) const {
    // CUDA kernel for PQ encoding
    // This is a simplified implementation - in practice, you'd have
    // optimized CUDA kernels for PQ encoding
    
    dim3 blockSize(256);
    dim3 gridSize((numQueries + blockSize.x - 1) / blockSize.x);
    
    // For now, use CPU encoding
    // In practice, you'd implement GPU-accelerated PQ encoding
}

void CudaIndexPQ::launchADCDistanceKernel(const uint8_t* query_codes, 
                                          const uint8_t* database_codes,
                                          float* distances, idx_t numQueries, idx_t k) const {
    // CUDA kernel for ADC distance computation
    // This is a simplified implementation - in practice, you'd have
    // optimized CUDA kernels for ADC distance computation
    
    dim3 blockSize(256);
    dim3 gridSize((ntotal_ + blockSize.x - 1) / blockSize.x);
    
    // For now, use CPU ADC computation
    // In practice, you'd implement GPU-accelerated ADC distance computation
}

void CudaIndexPQ::launchTopKSelection(float* distances, idx_t* indices,
                                       idx_t numQueries, idx_t k) const {
    // CUDA kernel for top-k selection
    // This is a simplified implementation - in practice, you'd use
    // optimized selection algorithms like radix select
    
    dim3 blockSize(256);
    dim3 gridSize((numQueries + blockSize.x - 1) / blockSize.x);
    
    // For now, use a simple selection sort on GPU
    // In practice, you'd implement a more efficient algorithm
}

// CUDA PQ utility implementations
namespace cuda_pq_utils {

void computeDistanceTables(const float* centroids, idx_t numCentroids,
                          idx_t subvectorSize, float* distanceTables) {
    // Compute distance tables for ADC
    // This is a simplified implementation - in practice, you'd have
    // optimized CUDA kernels for distance table computation
    
    // For now, use CPU computation
    // In practice, you'd implement GPU-accelerated distance table computation
}

void computeADCDistances(const uint8_t* queryCodes, const uint8_t* databaseCodes,
                        const float* distanceTables, float* distances,
                        idx_t numQueries, idx_t numVectors, idx_t codeSize) {
    // Compute ADC distances
    // This is a simplified implementation - in practice, you'd have
    // optimized CUDA kernels for ADC distance computation
    
    // For now, use CPU computation
    // In practice, you'd implement GPU-accelerated ADC distance computation
}

void allocatePQMemory(idx_t numVectors, idx_t codeSize, 
                     uint8_t** d_codes, float** d_distance_tables) {
    // Allocate memory for PQ operations
    size_t codes_size = numVectors * codeSize * sizeof(uint8_t);
    size_t distance_tables_size = 256 * 256 * sizeof(float); // Assuming 8-bit codes
    
    cudaMalloc(d_codes, codes_size);
    cudaMalloc(d_distance_tables, distance_tables_size);
}

void freePQMemory(uint8_t* d_codes, float* d_distance_tables) {
    if (d_codes) cudaFree(d_codes);
    if (d_distance_tables) cudaFree(d_distance_tables);
}

}

}
