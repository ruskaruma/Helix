#include"helix/core/threading.hpp"

namespace helix {

ThreadPool::ThreadPool(size_t numThreads) : stop_(false),activeTasks_(0)
{
    for(size_t i=0;i<numThreads;++i)
    {
        threads_.emplace_back([this]() { workerThread(); });
    }
}

ThreadPool::~ThreadPool()
{
    stop_=true;
    cv_.notify_all();
    
    for(auto& thread : threads_)
    {
        if(thread.joinable())
        {
            thread.join();
        }
    }
}

void ThreadPool::workerThread()
{
    while(!stop_)
    {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock,[this]() { return stop_ || !tasks_.empty(); });
            
            if(stop_ && tasks_.empty())
            {
                return;
            }
            
            if(!tasks_.empty())
            {
                task=std::move(tasks_.front());
                tasks_.pop();
                activeTasks_++;
            }
        }
        
        if(task)
        {
            task();
            activeTasks_--;
        }
    }
}

void ThreadPool::waitAll()
{
    while(activeTasks_>0 || !tasks_.empty())
    {
        std::this_thread::yield();
    }
}

void parallelFor(idx_t begin,idx_t end,std::function<void(idx_t)> func,size_t numThreads)
{
    if(numThreads==0)
    {
        numThreads=std::thread::hardware_concurrency();
    }
    
    if(numThreads==1 || (end-begin)<=1)
    {
        for(idx_t i=begin;i<end;++i)
        {
            func(i);
        }
        return;
    }
    
    ThreadPool pool(numThreads);
    idx_t chunkSize=(end-begin+numThreads-1)/numThreads;
    
    for(size_t chunk=0;chunk<numThreads;++chunk)
    {
        idx_t start=begin+chunk*chunkSize;
        idx_t finish=std::min(start+chunkSize,end);
        
        if(start>=finish)
        {
            break;
        }
        
        pool.enqueue([start,finish,&func]() {
            for(idx_t i=start;i<finish;++i)
            {
                func(i);
            }
        });
    }
    
    pool.waitAll();
}

}

