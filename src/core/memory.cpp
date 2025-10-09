#include"helix/core/memory.hpp"
#include<cstring>

namespace helix {

MemoryPool::MemoryPool(size_t blockSize,size_t maxBlocks)
    : blockSize_(blockSize),maxBlocks_(maxBlocks),allocated_(0),freeList_(nullptr)
{
}

MemoryPool::~MemoryPool()
{
}

void* MemoryPool::allocate()
{
    if(freeList_!=nullptr)
    {
        void* ptr=freeList_;
        freeList_=*reinterpret_cast<void**>(freeList_);
        return ptr;
    }
    
    if(allocated_>=maxBlocks_)
    {
        return nullptr;
    }
    
    void* ptr=nullptr;
    if(posix_memalign(&ptr,64,blockSize_)!=0)
    {
        return nullptr;
    }
    
    allocated_++;
    return ptr;
}

void MemoryPool::deallocate(void* ptr)
{
    if(ptr==nullptr)
    {
        return;
    }
    
    *reinterpret_cast<void**>(ptr)=freeList_;
    freeList_=ptr;
}

}

