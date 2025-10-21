#include "helix/cuda/cuda_simple.hpp"
#include "helix/common/utils.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace helix {

// CUDA utility implementations
namespace cuda_simple {

bool isAvailable() {
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

void synchronize() {
    cudaDeviceSynchronize();
}

void* allocate(size_t size) {
    void* ptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memory allocation failed");
    }
    return ptr;
}

void free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void memcpyHtoD(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memcpy HtoD failed");
    }
}

void memcpyDtoH(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memcpy DtoH failed");
    }
}

}

// CudaIndexFlatSimple implementation
CudaIndexFlatSimple::CudaIndexFlatSimple(dim_t dimension, MetricType metric)
    : dimension_(dimension), metric_(metric), ntotal_(0), trained_(false),
      d_vectors_(nullptr), d_queries_(nullptr), d_distances_(nullptr), 
      d_indices_(nullptr), gpu_memory_allocated_(0) {
    
    if (!cuda_simple::isAvailable()) {
        throw HelixException("CUDA not available");
    }
    
    allocateGpuMemory();
}

CudaIndexFlatSimple::~CudaIndexFlatSimple() {
    freeGpuMemory();
}

void CudaIndexFlatSimple::allocateGpuMemory() {
    // Allocate GPU memory for vectors (max 1M vectors)
    size_t maxVectors = 1000000;
    size_t vectorSize = maxVectors * dimension_ * sizeof(float);
    size_t querySize = dimension_ * sizeof(float);
    size_t distanceSize = maxVectors * sizeof(float);
    size_t indexSize = maxVectors * sizeof(idx_t);
    
    d_vectors_ = static_cast<float*>(cuda_simple::allocate(vectorSize));
    d_queries_ = static_cast<float*>(cuda_simple::allocate(querySize));
    d_distances_ = static_cast<float*>(cuda_simple::allocate(distanceSize));
    d_indices_ = static_cast<idx_t*>(cuda_simple::allocate(indexSize));
    
    gpu_memory_allocated_ = vectorSize + querySize + distanceSize + indexSize;
    
    // Initialize indices
    std::vector<idx_t> hostIndices(maxVectors);
    for (idx_t i = 0; i < maxVectors; ++i) {
        hostIndices[i] = i;
    }
    cuda_simple::memcpyHtoD(d_indices_, hostIndices.data(), indexSize);
}

void CudaIndexFlatSimple::freeGpuMemory() {
    cuda_simple::free(d_vectors_);
    cuda_simple::free(d_queries_);
    cuda_simple::free(d_distances_);
    cuda_simple::free(d_indices_);
    gpu_memory_allocated_ = 0;
}

void CudaIndexFlatSimple::train(const float* vectors, idx_t numVectors) {
    if (numVectors == 0) return;
    
    // Store vectors
    vectors_.resize(numVectors * dimension_);
    memcpy(vectors_.data(), vectors, numVectors * dimension_ * sizeof(float));
    
    trained_ = true;
}

void CudaIndexFlatSimple::add(const float* vectors, idx_t numVectors) {
    if (!trained_) {
        throw HelixException("Index must be trained before adding vectors");
    }
    
    if (numVectors == 0) return;
    
    // Append to existing vectors
    size_t oldSize = vectors_.size();
    vectors_.resize(oldSize + numVectors * dimension_);
    memcpy(vectors_.data() + oldSize, vectors, numVectors * dimension_ * sizeof(float));
    
    ntotal_ += numVectors;
}

void CudaIndexFlatSimple::syncToGpu() {
    if (ntotal_ > 0) {
        size_t vectorSize = ntotal_ * dimension_ * sizeof(float);
        cuda_simple::memcpyHtoD(d_vectors_, vectors_.data(), vectorSize);
    }
}

SearchResults CudaIndexFlatSimple::search(const float* query, idx_t k) const {
    if (k <= 0 || k > ntotal_) {
        throw HelixException("Invalid k value for search");
    }
    
    if (ntotal_ == 0) {
        return SearchResults();
    }
    
    // Sync data to GPU
    const_cast<CudaIndexFlatSimple*>(this)->syncToGpu();
    
    // Copy query to GPU
    cuda_simple::memcpyHtoD(d_queries_, query, dimension_ * sizeof(float));
    
    // Compute distances on GPU (simplified implementation)
    // In a real implementation, you'd have optimized CUDA kernels
    std::vector<float> distances(ntotal_);
    
    // For now, compute distances on CPU and copy to GPU
    for (idx_t i = 0; i < ntotal_; ++i) {
        float dist = 0.0f;
        for (dim_t d = 0; d < dimension_; ++d) {
            float diff = query[d] - vectors_[i * dimension_ + d];
            dist += diff * diff;
        }
        distances[i] = std::sqrt(dist);
    }
    
    // Copy distances to GPU
    cuda_simple::memcpyHtoD(d_distances_, distances.data(), ntotal_ * sizeof(float));
    
    // Find top-k (simplified implementation)
    std::vector<std::pair<float, idx_t>> distancePairs;
    for (idx_t i = 0; i < ntotal_; ++i) {
        distancePairs.push_back({distances[i], i});
    }
    
    std::sort(distancePairs.begin(), distancePairs.end());
    
    // Create results
    SearchResults result(k);
    
    for (idx_t i = 0; i < k; ++i) {
        result.results.emplace_back(distancePairs[i].second, distancePairs[i].first);
    }
    
    return result;
}

}
