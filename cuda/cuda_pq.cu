#include "helix/cuda/cuda_pq.hpp"
#include "helix/common/utils.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstring>

namespace helix
{
CudaIndexPQ::CudaIndexPQ(const IndexConfig& config) 
    : IndexPQ(config), d_codes_(nullptr), d_query_codes_(nullptr), 
      d_distances_(nullptr), d_indices_(nullptr), d_centroids_(nullptr),
      d_distance_tables_(nullptr), gpu_memory_allocated_(0),
      device_id_(0), gpu_data_valid_(false) {
    
    if (!cuda_utils::isCudaAvailable())
    {
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

void CudaIndexPQ::allocateGpuMemory()
{
    if (dimension_ == 0) return;
    
    size_t codes_size = ntotal_ * code_size_ * sizeof(uint8_t);
    size_t query_codes_size = code_size_ * sizeof(uint8_t);
    size_t distance_size = ntotal_ * sizeof(float);
    size_t index_size = ntotal_ * sizeof(idx_t);
    size_t centroids_size = quantizer_.getNumCentroids() * subvector_size_ * sizeof(float);
    size_t distance_tables_size = quantizer_.getNumCentroids() * quantizer_.getNumCentroids() * sizeof(float);
    cudaMalloc(&d_codes_, codes_size);
    cudaMalloc(&d_query_codes_, query_codes_size);
    cudaMalloc(&d_distances_, distance_size);
    cudaMalloc(&d_indices_, index_size);
    cudaMalloc(&d_centroids_, centroids_size);
    cudaMalloc(&d_distance_tables_, distance_tables_size);
    
    gpu_memory_allocated_ = codes_size + query_codes_size + distance_size + 
                           index_size + centroids_size + distance_tables_size;
    
    std::vector<idx_t> host_indices(ntotal_);
    for (idx_t i = 0; i < ntotal_; ++i) {
        host_indices[i] = i;
    }
    cudaMemcpy(d_indices_, host_indices.data(), index_size, cudaMemcpyHostToDevice);
}

void CudaIndexPQ::freeGpuMemory()
{
    if (d_codes_) { cudaFree(d_codes_); d_codes_ = nullptr; }
    if (d_query_codes_) { cudaFree(d_query_codes_); d_query_codes_ = nullptr; }
    if (d_distances_) { cudaFree(d_distances_); d_distances_ = nullptr; }
    if (d_indices_) { cudaFree(d_indices_); d_indices_ = nullptr; }
    if (d_centroids_) { cudaFree(d_centroids_); d_centroids_ = nullptr; }
    if (d_distance_tables_) { cudaFree(d_distance_tables_); d_distance_tables_ = nullptr; }
    gpu_memory_allocated_ = 0;
}

void CudaIndexPQ::syncToGpu()
{
    if (!gpu_data_valid_ && ntotal_ > 0)
    {
        size_t codes_size = ntotal_ * code_size_ * sizeof(uint8_t);
        cudaMemcpy(d_codes_, codes_.data(), codes_size, cudaMemcpyHostToDevice);
        
        size_t centroids_size = quantizer_.getNumCentroids() * subvector_size_ * sizeof(float);
        cudaMemcpy(d_centroids_, quantizer_.getCentroids().data(), 
                   centroids_size, cudaMemcpyHostToDevice);
        
        cuda_pq_utils::computeDistanceTables(d_centroids_, 
                                            quantizer_.getNumCentroids(),
                                            subvector_size_, 
                                            d_distance_tables_);
        
        gpu_data_valid_ = true;
    }
}

void CudaIndexPQ::syncFromGpu()
{
    //no sync needed for read-only operations
}

SearchResults CudaIndexPQ::search(const float* query, idx_t k) const
{
    if (k <= 0 || k > ntotal_) {
        throw HelixException("Invalid k value for search");
    }
    
    syncToGpu();
    
    //encoding query using PQ
    std::vector<uint8_t> query_codes(code_size_);
    quantizer_.encode(query, query_codes.data());
    cudaMemcpy(d_query_codes_, query_codes.data(), 
               code_size_ * sizeof(uint8_t), cudaMemcpyHostToDevice);
    launchADCDistanceKernel(d_query_codes_, d_codes_, d_distances_, 1, k);
    launchTopKSelection(d_distances_, d_indices_, 1, k);
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
    uint8_t* d_batch_query_codes;
    float* d_batch_distances;
    idx_t* d_batch_indices;

    size_t query_codes_size = numQueries * code_size_ * sizeof(uint8_t);
    size_t distance_size = numQueries * ntotal_ * sizeof(float);
    size_t index_size = numQueries * k * sizeof(idx_t);
    
    cudaMalloc(&d_batch_query_codes, query_codes_size);
    cudaMalloc(&d_batch_distances, distance_size);
    cudaMalloc(&d_batch_indices, index_size);
    std::vector<uint8_t> batch_query_codes(numQueries * code_size_);
    for (idx_t i = 0; i < numQueries; ++i) {
        quantizer_.encode(queries + i * dimension_, 
                         batch_query_codes.data() + i * code_size_);
    }
    //copy query codes to GPU
    cudaMemcpy(d_batch_query_codes, batch_query_codes.data(), 
               query_codes_size, cudaMemcpyHostToDevice);
    
    //launch batch ADC distance computation
    launchADCDistanceKernel(d_batch_query_codes, d_codes_, d_batch_distances, 
                           numQueries, k);
    
    //launch batch top-k selection
    launchTopKSelection(d_batch_distances, d_batch_indices, numQueries, k);
    
    //copy results back
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
    cudaFree(d_batch_query_codes);
    cudaFree(d_batch_distances);
    cudaFree(d_batch_indices);
}

void CudaIndexPQ::searchAsync(const float* query, idx_t k, cudaStream_t stream,
                              SearchResults* result) const {
    //implementation for async search
    *result = search(query, k);
}

void CudaIndexPQ::searchBatchAsync(const float* queries, idx_t numQueries, idx_t k,
                                   cudaStream_t stream, std::vector<SearchResults>* results) const {
    //async batch search
    searchBatch(queries, numQueries, k, *results);
}

void CudaIndexPQ::launchPQEncodeKernel(const float* queries, idx_t numQueries,
                                       uint8_t* codes) const {
    //kernel for PQ encoding
    
    dim3 blockSize(256);
    dim3 gridSize((numQueries + blockSize.x - 1) / blockSize.x);
    
}

void CudaIndexPQ::launchADCDistanceKernel(const uint8_t* query_codes, 
                                          const uint8_t* database_codes,
                                          float* distances, idx_t numQueries, idx_t k) const {
    //ADC distance computation
    
    dim3 blockSize(256);
    dim3 gridSize((ntotal_ + blockSize.x - 1) / blockSize.x);
    
}

void CudaIndexPQ::launchTopKSelection(float* distances, idx_t* indices,
                                       idx_t numQueries, idx_t k) const {
    //top-k selection
    
    dim3 blockSize(256);
    dim3 gridSize((numQueries + blockSize.x - 1) / blockSize.x);
    
}

//PQ utility implementations
namespace cuda_pq_utils {

void computeDistanceTables(const float* centroids, idx_t numCentroids,
                          idx_t subvectorSize, float* distanceTables) {
    //distance tables for ADC
    
}

void computeADCDistances(const uint8_t* queryCodes, const uint8_t* databaseCodes,
                        const float* distanceTables, float* distances,
                        idx_t numQueries, idx_t numVectors, idx_t codeSize) {
    //ADC distances
    
}

void allocatePQMemory(idx_t numVectors, idx_t codeSize, 
                     uint8_t** d_codes, float** d_distance_tables){
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
