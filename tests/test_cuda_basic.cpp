#include <gtest/gtest.h>
#include "helix/cuda/cuda_simple.hpp"
#include "helix/common/utils.hpp"
#include <vector>
#include <random>

class CudaBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!helix::cuda_simple::isAvailable()) {
            GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
        }
        
        index_ = std::make_unique<helix::CudaIndexFlatSimple>(128, helix::MetricType::L2);
        
        // Generate simple test data
        generateTestData();
    }
    
    void TearDown() override {
        index_.reset();
    }
    
    void generateTestData() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        // Generate 1000 vectors of dimension 128
        dataset_.resize(1000 * 128);
        for (size_t i = 0; i < dataset_.size(); ++i) {
            dataset_[i] = dis(gen);
        }
        
        // Generate query vector
        query_.resize(128);
        for (size_t i = 0; i < query_.size(); ++i) {
            query_[i] = dis(gen);
        }
    }
    
    std::unique_ptr<helix::CudaIndexFlatSimple> index_;
    std::vector<float> dataset_;
    std::vector<float> query_;
};

TEST_F(CudaBasicTest, Construction) {
    EXPECT_EQ(index_->dimension(), 128);
    EXPECT_EQ(index_->metric(), helix::MetricType::L2);
    EXPECT_EQ(index_->ntotal(), 0);
    EXPECT_FALSE(index_->isTrained());
}

TEST_F(CudaBasicTest, TrainAndAdd) {
    index_->train(dataset_.data(), 1000);
    EXPECT_TRUE(index_->isTrained());
    
    index_->add(dataset_.data(), 1000);
    EXPECT_EQ(index_->ntotal(), 1000);
}

TEST_F(CudaBasicTest, SearchBasic) {
    index_->train(dataset_.data(), 1000);
    index_->add(dataset_.data(), 1000);
    
    auto results = index_->search(query_.data(), 10);
    
    EXPECT_EQ(results.results.size(), 10);
    
    // Check that results are sorted by distance
    for (size_t i = 1; i < results.results.size(); ++i) {
        EXPECT_LE(results.results[i-1].distance, results.results[i].distance);
    }
}

TEST_F(CudaBasicTest, EdgeCases) {
    // Test with empty dataset
    helix::CudaIndexFlatSimple emptyIndex(128);
    EXPECT_EQ(emptyIndex.ntotal(), 0);
    
    // Test search with empty index
    EXPECT_THROW(emptyIndex.search(query_.data(), 10), std::exception);
    
    // Test with single vector
    std::vector<float> singleVector(128, 0.5f);
    index_->train(singleVector.data(), 1);
    index_->add(singleVector.data(), 1);
    
    auto results = index_->search(query_.data(), 1);
    EXPECT_EQ(results.results.size(), 1);
}

TEST_F(CudaBasicTest, ErrorHandling) {
    // Test invalid k values
    index_->train(dataset_.data(), 1000);
    index_->add(dataset_.data(), 1000);
    
    EXPECT_THROW(index_->search(query_.data(), 0), std::exception);
    EXPECT_THROW(index_->search(query_.data(), -1), std::exception);
    EXPECT_THROW(index_->search(query_.data(), 1001), std::exception);
    
    // Test null pointer
    EXPECT_THROW(index_->search(nullptr, 10), std::exception);
    
    // Test adding without training
    helix::CudaIndexFlatSimple untrainedIndex(128);
    EXPECT_THROW(untrainedIndex.add(dataset_.data(), 1000), std::exception);
}

TEST_F(CudaBasicTest, PerformanceTest) {
    index_->train(dataset_.data(), 1000);
    index_->add(dataset_.data(), 1000);
    
    // Measure search time
    auto start = std::chrono::high_resolution_clock::now();
    auto results = index_->search(query_.data(), 10);
    auto searchTime = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "CUDA search time: " << std::chrono::duration_cast<std::chrono::microseconds>(searchTime).count() 
              << " μs" << std::endl;
    
    // Results should be valid
    EXPECT_EQ(results.results.size(), 10);
}

TEST_F(CudaBasicTest, MemoryUsage) {
    index_->train(dataset_.data(), 1000);
    index_->add(dataset_.data(), 1000);
    
    // Check that GPU memory was allocated
    size_t freeMemory = helix::cuda_simple::getFreeMemory();
    EXPECT_GT(freeMemory, 0);
    
    std::cout << "Free GPU memory: " << freeMemory / (1024 * 1024) << " MB" << std::endl;
}

TEST_F(CudaBasicTest, DeviceManagement) {
    int deviceCount = helix::cuda_simple::getDeviceCount();
    EXPECT_GT(deviceCount, 0);
    
    std::cout << "CUDA device count: " << deviceCount << std::endl;
    
    // Test device switching
    if (deviceCount > 1) {
        helix::cuda_simple::setDevice(1);
        // Should not throw
    }
}

TEST_F(CudaBasicTest, BatchOperations) {
    index_->train(dataset_.data(), 1000);
    index_->add(dataset_.data(), 1000);
    
    // Test multiple searches
    for (int i = 0; i < 10; ++i) {
        auto results = index_->search(query_.data(), 5);
        EXPECT_EQ(results.results.size(), 5);
    }
}
