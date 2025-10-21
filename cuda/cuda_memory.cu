#include "helix/cuda/cuda_flat.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <memory>

namespace helix {
namespace cuda_memory {

// Memory pool for efficient GPU memory management
class CudaMemoryPool {
public:
    CudaMemoryPool(size_t initialSize = 1024 * 1024 * 1024) // 1GB initial
        : pool_size_(initialSize), allocated_size_(0) {
        cudaMalloc(&pool_ptr_, pool_size_);
        free_blocks_.push_back({pool_ptr_, pool_size_});
    }
    
    ~CudaMemoryPool() {
        if (pool_ptr_) {
            cudaFree(pool_ptr_);
        }
    }
    
    void* allocate(size_t size) {
        // Align size to 256 bytes for optimal performance
        size_t aligned_size = (size + 255) & ~255;
        
        // Find best fit block
        auto it = std::find_if(free_blocks_.begin(), free_blocks_.end(),
                              [aligned_size](const Block& block) {
                                  return block.size >= aligned_size;
                              });
        
        if (it == free_blocks_.end()) {
            // No suitable block found, need to expand pool
            expandPool(aligned_size);
            it = free_blocks_.begin();
        }
        
        void* ptr = it->ptr;
        size_t remaining = it->size - aligned_size;
        
        if (remaining > 0) {
            // Split block
            it->ptr = static_cast<char*>(it->ptr) + aligned_size;
            it->size = remaining;
        } else {
            // Remove block
            free_blocks_.erase(it);
        }
        
        allocated_size_ += aligned_size;
        return ptr;
    }
    
    void deallocate(void* ptr, size_t size) {
        if (!ptr) return;
        
        // Align size
        size_t aligned_size = (size + 255) & ~255;
        
        // Add to free blocks
        free_blocks_.push_back({ptr, aligned_size});
        allocated_size_ -= aligned_size;
        
        // Merge adjacent blocks
        mergeBlocks();
    }
    
    size_t getTotalSize() const { return pool_size_; }
    size_t getAllocatedSize() const { return allocated_size_; }
    size_t getFreeSize() const { return pool_size_ - allocated_size_; }
    
private:
    struct Block {
        void* ptr;
        size_t size;
    };
    
    void* pool_ptr_;
    size_t pool_size_;
    size_t allocated_size_;
    std::vector<Block> free_blocks_;
    
    void expandPool(size_t requiredSize) {
        // Double the pool size
        size_t new_size = std::max(pool_size_ * 2, pool_size_ + requiredSize);
        
        void* new_pool;
        cudaMalloc(&new_pool, new_size);
        
        // Copy existing data
        cudaMemcpy(new_pool, pool_ptr_, allocated_size_, cudaMemcpyDeviceToDevice);
        
        // Free old pool
        cudaFree(pool_ptr_);
        
        // Update pool
        pool_ptr_ = new_pool;
        pool_size_ = new_size;
        
        // Add new free block
        free_blocks_.push_back({static_cast<char*>(new_pool) + allocated_size_, 
                               new_size - allocated_size_});
    }
    
    void mergeBlocks() {
        // Sort blocks by address
        std::sort(free_blocks_.begin(), free_blocks_.end(),
                 [](const Block& a, const Block& b) {
                     return a.ptr < b.ptr;
                 });
        
        // Merge adjacent blocks
        for (auto it = free_blocks_.begin(); it != free_blocks_.end() - 1; ) {
            auto next = it + 1;
            if (static_cast<char*>(it->ptr) + it->size == next->ptr) {
                // Merge blocks
                it->size += next->size;
                free_blocks_.erase(next);
            } else {
                ++it;
            }
        }
    }
};

// Global memory pool instance
static std::unique_ptr<CudaMemoryPool> g_memory_pool;

void initializeMemoryPool(size_t initialSize) {
    g_memory_pool = std::make_unique<CudaMemoryPool>(initialSize);
}

void cleanupMemoryPool() {
    g_memory_pool.reset();
}

void* allocate(size_t size) {
    if (!g_memory_pool) {
        initializeMemoryPool();
    }
    return g_memory_pool->allocate(size);
}

void deallocate(void* ptr, size_t size) {
    if (g_memory_pool) {
        g_memory_pool->deallocate(ptr, size);
    }
}

// Memory statistics
struct MemoryStats {
    size_t total_allocated;
    size_t peak_allocated;
    size_t current_allocated;
    size_t allocation_count;
    size_t deallocation_count;
};

static MemoryStats g_stats = {0, 0, 0, 0, 0};

void* allocateWithStats(size_t size) {
    void* ptr = allocate(size);
    if (ptr) {
        g_stats.current_allocated += size;
        g_stats.total_allocated += size;
        g_stats.peak_allocated = std::max(g_stats.peak_allocated, g_stats.current_allocated);
        g_stats.allocation_count++;
    }
    return ptr;
}

void deallocateWithStats(void* ptr, size_t size) {
    deallocate(ptr, size);
    if (ptr) {
        g_stats.current_allocated -= size;
        g_stats.deallocation_count++;
    }
}

MemoryStats getMemoryStats() {
    return g_stats;
}

void resetMemoryStats() {
    g_stats = {0, 0, 0, 0, 0};
}

// Memory alignment utilities
size_t alignSize(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

bool isAligned(void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// Memory copying utilities
void copyHtoD(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memcpy HtoD failed: " + std::string(cudaGetErrorString(error)));
    }
}

void copyDtoH(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memcpy DtoH failed: " + std::string(cudaGetErrorString(error)));
    }
}

void copyDtoD(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memcpy DtoD failed: " + std::string(cudaGetErrorString(error)));
    }
}

void copyAsync(void* dst, const void* src, size_t size, cudaStream_t stream) {
    cudaError_t error = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memcpy async failed: " + std::string(cudaGetErrorString(error)));
    }
}

// Memory initialization utilities
void memset(void* ptr, int value, size_t size) {
    cudaError_t error = cudaMemset(ptr, value, size);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memset failed: " + std::string(cudaGetErrorString(error)));
    }
}

void memsetAsync(void* ptr, int value, size_t size, cudaStream_t stream) {
    cudaError_t error = cudaMemsetAsync(ptr, value, size, stream);
    if (error != cudaSuccess) {
        throw HelixException("CUDA memset async failed: " + std::string(cudaGetErrorString(error)));
    }
}

// Memory prefetching utilities
void prefetchToDevice(void* ptr, size_t size) {
    cudaError_t error = cudaMemPrefetchAsync(ptr, size, 0, 0); // Default stream
    if (error != cudaSuccess) {
        throw HelixException("CUDA prefetch failed: " + std::string(cudaGetErrorString(error)));
    }
}

void prefetchToHost(void* ptr, size_t size) {
    cudaError_t error = cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, 0);
    if (error != cudaSuccess) {
        throw HelixException("CUDA prefetch to host failed: " + std::string(cudaGetErrorString(error)));
    }
}

}
}
