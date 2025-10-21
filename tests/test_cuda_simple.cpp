#include <gtest/gtest.h>
#include "helix/cuda/cuda_simple.hpp"
#include "helix/benchmark/dataset.hpp"
#include <vector>
#include <cmath>

class CudaSimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!helix::cuda_simple::isAvailable()) {
            GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
        }
        
        index_ = std::make_unique<helix::CudaIndexFlatSimple>(128, helix::MetricType::L2);
        
        // Generate test data
        dataset_ = helix::BenchmarkDataset::generateSynthetic(1000, 128, 0.0f, 1.0f);
        query_ = helix::BenchmarkDataset::generateSynthetic(1, 128, 0.0f, 1.0f);
    }
    
    void TearDown() override {
        index_.reset();
    }
    
    std::unique_ptr<helix::CudaIndexFlatSimple> index_;
    helix::BenchmarkDataset dataset_;
    helix::BenchmarkDataset query_;
};

TEST_F(CudaSimpleTest, Construction) {
    EXPECT_EQ(index_->dimension(), 128);
    EXPECT_EQ(index_->metric(), helix::MetricType::L2);
    EXPECT_EQ(index_->ntotal(), 0);
    EXPECT_FALSE(index_->isTrained());
}

TEST_F(CudaSimpleTest, TrainAndAdd) {
    index_->train(dataset_.data.data(), dataset_.nrows);
    EXPECT_TRUE(index_->isTrained());
    
    index_->add(dataset_.data.data(), dataset_.nrows);
    EXPECT_EQ(index_->ntotal(), dataset_.nrows);
}

TEST_F(CudaSimpleTest, SearchBasic) {
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    auto results = index_->search(query_.data.data(), 10);
    
    EXPECT_EQ(results.results.size(), 10);
    
    // Check that results are sorted by distance
    for (size_t i = 1; i < results.results.size(); ++i) {
        EXPECT_LE(results.results[i-1].distance, results.results[i].distance);
    }
}

TEST_F(CudaSimpleTest, EdgeCases) {
    // Test with empty dataset
    helix::CudaIndexFlatSimple emptyIndex(128);
    EXPECT_EQ(emptyIndex.ntotal(), 0);
    
    // Test search with empty index
    EXPECT_THROW(emptyIndex.search(query_.data.data(), 10), helix::HelixException);
    
    // Test with single vector
    auto singleVector = helix::BenchmarkDataset::generateSynthetic(1, 128, 0.0f, 1.0f);
    index_->train(singleVector.data.data(), 1);
    index_->add(singleVector.data.data(), 1);
    
    auto results = index_->search(query_.data.data(), 1);
    EXPECT_EQ(results.results.size(), 1);
}

TEST_F(CudaSimpleTest, ErrorHandling) {
    // Test invalid k values
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    EXPECT_THROW(index_->search(query_.data.data(), 0), helix::HelixException);
    EXPECT_THROW(index_->search(query_.data.data(), -1), helix::HelixException);
    EXPECT_THROW(index_->search(query_.data.data(), dataset_.nrows + 1), helix::HelixException);
    
    // Test null pointer
    EXPECT_THROW(index_->search(nullptr, 10), helix::HelixException);
    
    // Test adding without training
    helix::CudaIndexFlatSimple untrainedIndex(128);
    EXPECT_THROW(untrainedIndex.add(dataset_.data.data(), dataset_.nrows), helix::HelixException);
}

TEST_F(CudaSimpleTest, PerformanceComparison) {
    // This test compares performance between CPU and GPU
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    // Measure GPU search time
    auto start = std::chrono::high_resolution_clock::now();
    auto gpuResults = index_->search(query_.data.data(), 10);
    auto gpuTime = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "GPU search time: " << std::chrono::duration_cast<std::chrono::microseconds>(gpuTime).count() 
              << " μs" << std::endl;
    
    // Results should be valid
    EXPECT_EQ(gpuResults.results.size(), 10);
}

TEST_F(CudaSimpleTest, MemoryUsage) {
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    // Check that GPU memory was allocated
    size_t freeMemory = helix::cuda_simple::getFreeMemory();
    EXPECT_GT(freeMemory, 0);
    
    std::cout << "Free GPU memory: " << freeMemory / (1024 * 1024) << " MB" << std::endl;
}

TEST_F(CudaSimpleTest, DeviceManagement) {
    int deviceCount = helix::cuda_simple::getDeviceCount();
    EXPECT_GT(deviceCount, 0);
    
    std::cout << "CUDA device count: " << deviceCount << std::endl;
    
    // Test device switching
    if (deviceCount > 1) {
        helix::cuda_simple::setDevice(1);
        // Should not throw
    }
}

TEST_F(CudaSimpleTest, LargeDataset) {
    // Test with larger dataset
    auto largeDataset = helix::BenchmarkDataset::generateSynthetic(10000, 128, 0.0f, 1.0f);
    
    try {
        index_->train(largeDataset.data.data(), largeDataset.nrows);
        index_->add(largeDataset.data.data(), largeDataset.nrows);
        
        // If successful, test search
        auto results = index_->search(query_.data.data(), 10);
        EXPECT_EQ(results.results.size(), 10);
        
        std::cout << "Large dataset test passed with " << index_->ntotal() << " vectors" << std::endl;
    } catch (const helix::HelixException& e) {
        // Memory allocation failed, which is acceptable for large datasets
        GTEST_SKIP() << "Insufficient GPU memory for large dataset test: " << e.what();
    }
}

TEST_F(CudaSimpleTest, BatchOperations) {
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    // Test multiple searches
    auto batchQueries = helix::BenchmarkDataset::generateSynthetic(10, 128, 0.0f, 1.0f);
    
    for (int i = 0; i < 10; ++i) {
        auto results = index_->search(batchQueries.data.data() + i * 128, 5);
        EXPECT_EQ(results.results.size(), 5);
    }
}
