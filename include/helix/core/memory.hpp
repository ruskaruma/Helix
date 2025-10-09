#pragma once

#include<cstddef>
#include<memory>
#include<cstdlib>

namespace helix {

template<typename T>
class AlignedAllocator {
public:
    using value_type=T;
    static constexpr size_t alignment=64;
    
    AlignedAllocator()=default;
    
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U>&) {}
    
    T* allocate(size_t n)
    {
        void* ptr=nullptr;
        if(posix_memalign(&ptr,alignment,n*sizeof(T))!=0)
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr,size_t)
    {
        free(ptr);
    }
};

template<typename T,typename U>
bool operator==(const AlignedAllocator<T>&,const AlignedAllocator<U>&)
{
    return true;
}

template<typename T,typename U>
bool operator!=(const AlignedAllocator<T>&,const AlignedAllocator<U>&)
{
    return false;
}

class MemoryPool {
public:
    MemoryPool(size_t blockSize,size_t maxBlocks);
    ~MemoryPool();
    
    void* allocate();
    void deallocate(void* ptr);
    
    size_t allocated() const { return allocated_; }
    size_t capacity() const { return maxBlocks_; }
    
private:
    size_t blockSize_;
    size_t maxBlocks_;
    size_t allocated_;
    void* freeList_;
};

}

