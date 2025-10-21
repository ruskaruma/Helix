#include "helix/cuda/cuda_flat.hpp"
#include "helix/common/utils.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstring>

namespace helix {

CudaIndexFlat::CudaIndexFlat(const IndexConfig& config) 
    : IndexFlat(config), d_vectors_(nullptr), d_queries_(nullptr), 
      d_distances_(nullptr), d_indices_(nullptr), gpu_memory_allocated_(0),
      device_id_(0), gpu_data_valid_(false) {
    
    if (!cuda_utils::isCudaAvailable()) {
        throw HelixException("CUDA not available on this system");
    }
    
    // Initialize CUDA
    cudaSetDevice(device_id_);
    cublasCreate(&cublas_handle_);
    cudaStreamCreate(&stream_);
    
    allocateGpuMemory();
}

CudaIndexFlat::~CudaIndexFlat() {
    freeGpuMemory();
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void CudaIndexFlat::setDevice(int deviceId) {
    if (deviceId < 0 || deviceId >= cuda_utils::getDeviceCount()) {
        throw HelixException("Invalid CUDA device ID");
    }
    
    device_id_ = deviceId;
    cudaSetDevice(device_id_);
    
    // Reallocate memory on new device
    freeGpuMemory();
    allocateGpuMemory();
    gpu_data_valid_ = false;
}

int CudaIndexFlat::getDevice() const {
    return device_id_;
}

size_t CudaIndexFlat::getGpuMemoryUsage() const {
    return gpu_memory_allocated_;
}

void CudaIndexFlat::allocateGpuMemory() {
    if (dimension_ == 0) return;
    
    // Calculate memory requirements
    size_t vector_size = ntotal_ * dimension_ * sizeof(float);
    size_t query_size = dimension_ * sizeof(float);
    size_t distance_size = ntotal_ * sizeof(float);
    size_t index_size = ntotal_ * sizeof(idx_t);
    
    // Allocate GPU memory
    cudaMalloc(&d_vectors_, vector_size);
    cudaMalloc(&d_queries_, query_size);
    cudaMalloc(&d_distances_, distance_size);
    cudaMalloc(&d_indices_, index_size);
    
    gpu_memory_allocated_ = vector_size + query_size + distance_size + index_size;
    
    // Initialize indices
    std::vector<idx_t> host_indices(ntotal_);
    for (idx_t i = 0; i < ntotal_; ++i) {
        host_indices[i] = i;
    }
    cudaMemcpy(d_indices_, host_indices.data(), index_size, cudaMemcpyHostToDevice);
}

void CudaIndexFlat::freeGpuMemory() {
    if (d_vectors_) { cudaFree(d_vectors_); d_vectors_ = nullptr; }
    if (d_queries_) { cudaFree(d_queries_); d_queries_ = nullptr; }
    if (d_distances_) { cudaFree(d_distances_); d_distances_ = nullptr; }
    if (d_indices_) { cudaFree(d_indices_); d_indices_ = nullptr; }
    gpu_memory_allocated_ = 0;
}

void CudaIndexFlat::syncToGpu() {
    if (!gpu_data_valid_ && ntotal_ > 0) {
        size_t vector_size = ntotal_ * dimension_ * sizeof(float);
        cudaMemcpy(d_vectors_, vectors_.data(), vector_size, cudaMemcpyHostToDevice);
        gpu_data_valid_ = true;
    }
}

void CudaIndexFlat::syncFromGpu() {
    // For read-only operations, no sync needed
}

SearchResults CudaIndexFlat::search(const float* query, idx_t k) const {
    if (k <= 0 || k > ntotal_) {
        throw HelixException("Invalid k value for search");
    }
    
    syncToGpu();
    
    // Copy query to GPU
    cudaMemcpy(d_queries_, query, dimension_ * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch distance computation kernel
    launchDistanceKernel(d_queries_, 1, d_distances_, k);
    
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

void CudaIndexFlat::searchBatch(const float* queries, idx_t numQueries, idx_t k,
                                std::vector<SearchResults>& results) const {
    if (numQueries == 0) return;
    
    syncToGpu();
    
    // Allocate temporary GPU memory for batch
    float* d_batch_queries;
    float* d_batch_distances;
    idx_t* d_batch_indices;
    
    size_t query_size = numQueries * dimension_ * sizeof(float);
    size_t distance_size = numQueries * ntotal_ * sizeof(float);
    size_t index_size = numQueries * k * sizeof(idx_t);
    
    cudaMalloc(&d_batch_queries, query_size);
    cudaMalloc(&d_batch_distances, distance_size);
    cudaMalloc(&d_batch_indices, index_size);
    
    // Copy queries to GPU
    cudaMemcpy(d_batch_queries, queries, query_size, cudaMemcpyHostToDevice);
    
    // Launch batch distance computation
    launchDistanceKernel(d_batch_queries, numQueries, d_batch_distances, k);
    
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
    cudaFree(d_batch_queries);
    cudaFree(d_batch_distances);
    cudaFree(d_batch_indices);
}

void CudaIndexFlat::searchAsync(const float* query, idx_t k, cudaStream_t stream,
                                SearchResults* result) const {
    // Implementation for async search
    // This would use the provided stream for non-blocking execution
    // For now, fall back to synchronous search
    *result = search(query, k);
}

void CudaIndexFlat::searchBatchAsync(const float* queries, idx_t numQueries, idx_t k,
                                     cudaStream_t stream, std::vector<SearchResults>* results) const {
    // Implementation for async batch search
    // This would use the provided stream for non-blocking execution
    // For now, fall back to synchronous batch search
    searchBatch(queries, numQueries, k, *results);
}

void CudaIndexFlat::launchDistanceKernel(const float* queries, idx_t numQueries,
                                         float* distances, idx_t k) const {
    // CUDA kernel launch for distance computation
    // This is a simplified implementation - in practice, you'd have
    // optimized CUDA kernels for different distance metrics
    
    dim3 blockSize(256);
    dim3 gridSize((ntotal_ + blockSize.x - 1) / blockSize.x);
    
    // For L2 distance, use CUBLAS
    if (metric_ == MetricType::L2) {
        const float alpha = -2.0f;
        const float beta = 0.0f;
        
        // Compute -2 * queries^T * vectors
        cublasSgemm(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                   ntotal_, numQueries, dimension_,
                   &alpha, d_vectors_, dimension_,
                   queries, dimension_,
                   &beta, distances, ntotal_);
        
        // Add ||query||^2 to each row (simplified)
        // In practice, you'd have a more sophisticated kernel
    }
}

void CudaIndexFlat::launchTopKSelection(float* distances, idx_t* indices,
                                        idx_t numQueries, idx_t k) const {
    // CUDA kernel for top-k selection
    // This is a simplified implementation - in practice, you'd use
    // optimized selection algorithms like radix select
    
    dim3 blockSize(256);
    dim3 gridSize((numQueries + blockSize.x - 1) / blockSize.x);
    
    // For now, use a simple selection sort on GPU
    // In practice, you'd implement a more efficient algorithm
}

// CUDA utility implementations
namespace cuda_utils {

bool isCudaAvailable() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return error == cudaSuccess && deviceCount > 0;
}

int getDeviceCount() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

void setDevice(int deviceId) {
    cudaSetDevice(deviceId);
}

size_t getFreeMemory() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

size_t getTotalMemory() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return total;
}

void synchronize() {
    cudaDeviceSynchronize();
}

void* allocate(size_t size) {
    void* ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void free(void* ptr) {
    cudaFree(ptr);
}

void memcpyHtoD(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void memcpyDtoH(void* dst, const void* src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void memcpyAsync(void* dst, const void* src, size_t size, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}

}

}
