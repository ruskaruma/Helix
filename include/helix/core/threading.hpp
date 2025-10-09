#pragma once

#include"helix/common/types.hpp"
#include<thread>
#include<vector>
#include<functional>
#include<mutex>
#include<condition_variable>
#include<queue>
#include<atomic>

namespace helix {

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads);
    ~ThreadPool();
    
    template<typename Func>
    void enqueue(Func&& task)
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            tasks_.emplace(std::forward<Func>(task));
        }
        cv_.notify_one();
    }
    
    void waitAll();
    size_t numThreads() const { return threads_.size(); }
    
private:
    void workerThread();
    
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_;
    std::atomic<int> activeTasks_;
};

void parallelFor(idx_t begin,idx_t end,std::function<void(idx_t)> func,size_t numThreads=0);

}

