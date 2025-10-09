#include<gtest/gtest.h>
#include"helix/core/threading.hpp"
#include<atomic>

TEST(ThreadingTest,ThreadPoolBasic) {
    helix::ThreadPool pool(4);
    
    std::atomic<int> counter(0);
    
    for(int i=0;i<100;++i)
    {
        pool.enqueue([&counter]() {
            counter++;
        });
    }
    
    pool.waitAll();
    EXPECT_EQ(counter,100);
}

TEST(ThreadingTest,ParallelFor) {
    std::vector<int> data(1000,0);
    
    helix::parallelFor(0,1000,[&data](helix::idx_t i) {
        data[i]=i*2;
    },4);
    
    for(int i=0;i<1000;++i)
    {
        EXPECT_EQ(data[i],i*2);
    }
}

TEST(ThreadingTest,ParallelForSingleThread) {
    std::vector<int> data(10,0);
    
    helix::parallelFor(0,10,[&data](helix::idx_t i) {
        data[i]=i;
    },1);
    
    for(int i=0;i<10;++i)
    {
        EXPECT_EQ(data[i],i);
    }
}

