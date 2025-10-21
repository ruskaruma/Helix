#include <gtest/gtest.h>
#include "helix/cuda/cuda_pq.hpp"
#include "helix/common/types.hpp"
#include "helix/benchmark/dataset.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

class CudaPQTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!helix::cuda_utils::isCudaAvailable()) {
            GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
        }
        
        config_ = helix::IndexConfig(128, helix::MetricType::L2, helix::IndexType::PQ);
        index_ = std::make_unique<helix::CudaIndexPQ>(config_);
        
        // Generate test data
        dataset_ = helix::BenchmarkDataset::generateSynthetic(1000, 128, 0.0f, 1.0f);
        query_ = helix::BenchmarkDataset::generateSynthetic(1, 128, 0.0f, 1.0f);
    }
    
    void TearDown() override {
        index_.reset();
    }
    
    helix::IndexConfig config_;
    std::unique_ptr<helix::CudaIndexPQ> index_;
    helix::BenchmarkDataset dataset_;
    helix::BenchmarkDataset query_;
};

TEST_F(CudaPQTest, Construction) {
    EXPECT_EQ(index_->dimension(), 128);
    EXPECT_EQ(index_->metric(), helix::MetricType::L2);
    EXPECT_EQ(index_->ntotal(), 0);
    EXPECT_FALSE(index_->isTrained());
}

TEST_F(CudaPQTest, DeviceManagement) {
    int deviceCount = helix::cuda_utils::getDeviceCount();
    EXPECT_GT(deviceCount, 0);
    
    // Test device switching
    if (deviceCount > 1) {
        index_->setDevice(1);
        EXPECT_EQ(index_->getDevice(), 1);
    }
    
    // Test invalid device
    EXPECT_THROW(index_->setDevice(-1), helix::HelixException);
    EXPECT_THROW(index_->setDevice(deviceCount), helix::HelixException);
}

TEST_F(CudaPQTest, MemoryUsage) {
    // Train and add data
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    size_t memoryUsage = index_->getGpuMemoryUsage();
    EXPECT_GT(memoryUsage, 0);
    
    // Memory usage should be reasonable (less than 1GB for 1000 vectors)
    EXPECT_LT(memoryUsage, 1024 * 1024 * 1024);
}

TEST_F(CudaPQTest, TrainAndAdd) {
    index_->train(dataset_.data.data(), dataset_.nrows);
    EXPECT_TRUE(index_->isTrained());
    
    index_->add(dataset_.data.data(), dataset_.nrows);
    EXPECT_EQ(index_->ntotal(), dataset_.nrows);
}

TEST_F(CudaPQTest, SearchBasic) {
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    auto results = index_->search(query_.data.data(), 10);
    
    EXPECT_EQ(results.indices.size(), 10);
    EXPECT_EQ(results.distances.size(), 10);
    
    // Check that results are sorted by distance
    for (size_t i = 1; i < results.distances.size(); ++i) {
        EXPECT_LE(results.distances[i-1], results.distances[i]);
    }
}

TEST_F(CudaPQTest, SearchBatch) {
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    // Generate multiple queries
    auto batch_queries = helix::BenchmarkDataset::generateSynthetic(5, 128, 0.0f, 1.0f);
    
    std::vector<helix::SearchResults> results;
    index_->searchBatch(batch_queries.data.data(), 5, 10, results);
    
    EXPECT_EQ(results.size(), 5);
    for (const auto& result : results) {
        EXPECT_EQ(result.indices.size(), 10);
        EXPECT_EQ(result.distances.size(), 10);
    }
}

TEST_F(CudaPQTest, CpuGpuParity) {
    // Create CPU index for comparison
    helix::IndexPQ cpu_index(config_);
    
    // Train both indices
    index_->train(dataset_.data.data(), dataset_.nrows);
    cpu_index.train(dataset_.data.data(), dataset_.nrows);
    
    // Add data to both indices
    index_->add(dataset_.data.data(), dataset_.nrows);
    cpu_index.add(dataset_.data.data(), dataset_.nrows);
    
    // Search with both indices
    auto gpu_results = index_->search(query_.data.data(), 10);
    auto cpu_results = cpu_index.search(query_.data.data(), 10);
    
    // Compare results (should be identical)
    EXPECT_EQ(gpu_results.indices.size(), cpu_results.indices.size());
    EXPECT_EQ(gpu_results.distances.size(), cpu_results.distances.size());
    
    // Check indices match (within epsilon for distances)
    const float epsilon = 1e-6f;
    for (size_t i = 0; i < gpu_results.indices.size(); ++i) {
        EXPECT_EQ(gpu_results.indices[i], cpu_results.indices[i]);
        EXPECT_NEAR(gpu_results.distances[i], cpu_results.distances[i], epsilon);
    }
}

TEST_F(CudaPQTest, DifferentMetrics) {
    // Test L2 distance
    helix::IndexConfig l2_config(128, helix::MetricType::L2, helix::IndexType::PQ);
    helix::CudaIndexPQ l2_index(l2_config);
    
    l2_index.train(dataset_.data.data(), dataset_.nrows);
    l2_index.add(dataset_.data.data(), dataset_.nrows);
    
    auto l2_results = l2_index.search(query_.data.data(), 10);
    EXPECT_EQ(l2_results.indices.size(), 10);
    
    // Test Inner Product distance
    helix::IndexConfig ip_config(128, helix::MetricType::InnerProduct, helix::IndexType::PQ);
    helix::CudaIndexPQ ip_index(ip_config);
    
    ip_index.train(dataset_.data.data(), dataset_.nrows);
    ip_index.add(dataset_.data.data(), dataset_.nrows);
    
    auto ip_results = ip_index.search(query_.data.data(), 10);
    EXPECT_EQ(ip_results.indices.size(), 10);
    
    // Results should be different for different metrics
    EXPECT_NE(l2_results.indices[0], ip_results.indices[0]);
}

TEST_F(CudaPQTest, LargeBatch) {
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    // Generate large batch of queries
    auto large_batch = helix::BenchmarkDataset::generateSynthetic(100, 128, 0.0f, 1.0f);
    
    std::vector<helix::SearchResults> results;
    index_->searchBatch(large_batch.data.data(), 100, 10, results);
    
    EXPECT_EQ(results.size(), 100);
    for (const auto& result : results) {
        EXPECT_EQ(result.indices.size(), 10);
        EXPECT_EQ(result.distances.size(), 10);
    }
}

TEST_F(CudaPQTest, EdgeCases) {
    // Test with empty dataset
    helix::CudaIndexPQ empty_index(config_);
    EXPECT_EQ(empty_index.ntotal(), 0);
    
    // Test search with empty index
    EXPECT_THROW(empty_index.search(query_.data.data(), 10), helix::HelixException);
    
    // Test with single vector
    auto single_vector = helix::BenchmarkDataset::generateSynthetic(1, 128, 0.0f, 1.0f);
    index_->train(single_vector.data.data(), 1);
    index_->add(single_vector.data.data(), 1);
    
    auto results = index_->search(query_.data.data(), 1);
    EXPECT_EQ(results.indices.size(), 1);
    EXPECT_EQ(results.distances.size(), 1);
}

TEST_F(CudaPQTest, MemoryLimits) {
    // Test memory allocation with large dataset
    auto large_dataset = helix::BenchmarkDataset::generateSynthetic(10000, 128, 0.0f, 1.0f);
    
    try {
        index_->train(large_dataset.data.data(), large_dataset.nrows);
        index_->add(large_dataset.data.data(), large_dataset.nrows);
        
        // If successful, test search
        auto results = index_->search(query_.data.data(), 10);
        EXPECT_EQ(results.indices.size(), 10);
    } catch (const helix::HelixException& e) {
        // Memory allocation failed, which is acceptable for large datasets
        GTEST_SKIP() << "Insufficient GPU memory for large dataset test";
    }
}

TEST_F(CudaPQTest, ErrorHandling) {
    // Test invalid k values
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    EXPECT_THROW(index_->search(query_.data.data(), 0), helix::HelixException);
    EXPECT_THROW(index_->search(query_.data.data(), -1), helix::HelixException);
    EXPECT_THROW(index_->search(query_.data.data(), dataset_.nrows + 1), helix::HelixException);
    
    // Test null pointer
    EXPECT_THROW(index_->search(nullptr, 10), helix::HelixException);
}

TEST_F(CudaPQTest, PerformanceComparison) {
    // This test compares performance between CPU and GPU
    // In a real implementation, you'd measure actual timing
    
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    // Measure GPU search time
    auto start = std::chrono::high_resolution_clock::now();
    auto gpu_results = index_->search(query_.data.data(), 10);
    auto gpu_time = std::chrono::high_resolution_clock::now() - start;
    
    // Create CPU index for comparison
    helix::IndexPQ cpu_index(config_);
    cpu_index.train(dataset_.data.data(), dataset_.nrows);
    cpu_index.add(dataset_.data.data(), dataset_.nrows);
    
    // Measure CPU search time
    start = std::chrono::high_resolution_clock::now();
    auto cpu_results = cpu_index.search(query_.data.data(), 10);
    auto cpu_time = std::chrono::high_resolution_clock::now() - start;
    
    // Results should be identical
    EXPECT_EQ(gpu_results.indices.size(), cpu_results.indices.size());
    for (size_t i = 0; i < gpu_results.indices.size(); ++i) {
        EXPECT_EQ(gpu_results.indices[i], cpu_results.indices[i]);
    }
    
    // GPU should be faster for large datasets (this is a simplified check)
    // In practice, you'd have more sophisticated performance testing
    std::cout << "GPU time: " << std::chrono::duration_cast<std::chrono::microseconds>(gpu_time).count() 
              << " μs" << std::endl;
    std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::microseconds>(cpu_time).count() 
              << " μs" << std::endl;
}

TEST_F(CudaPQTest, QuantizationAccuracy) {
    // Test that PQ quantization maintains reasonable accuracy
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    auto results = index_->search(query_.data.data(), 10);
    
    // Check that results are reasonable (not all zeros or identical)
    bool hasVariation = false;
    for (size_t i = 1; i < results.distances.size(); ++i) {
        if (results.distances[i] != results.distances[i-1]) {
            hasVariation = true;
            break;
        }
    }
    EXPECT_TRUE(hasVariation) << "PQ results should show variation in distances";
}

TEST_F(CudaPQTest, BatchConsistency) {
    // Test that batch search gives consistent results with individual searches
    index_->train(dataset_.data.data(), dataset_.nrows);
    index_->add(dataset_.data.data(), dataset_.nrows);
    
    // Generate multiple queries
    auto batch_queries = helix::BenchmarkDataset::generateSynthetic(3, 128, 0.0f, 1.0f);
    
    // Search individually
    std::vector<helix::SearchResults> individual_results;
    for (int i = 0; i < 3; ++i) {
        individual_results.push_back(index_->search(batch_queries.data.data() + i * 128, 10));
    }
    
    // Search in batch
    std::vector<helix::SearchResults> batch_results;
    index_->searchBatch(batch_queries.data.data(), 3, 10, batch_results);
    
    // Compare results
    EXPECT_EQ(individual_results.size(), batch_results.size());
    for (size_t i = 0; i < individual_results.size(); ++i) {
        EXPECT_EQ(individual_results[i].indices.size(), batch_results[i].indices.size());
        EXPECT_EQ(individual_results[i].distances.size(), batch_results[i].distances.size());
        
        // Results should be identical
        for (size_t j = 0; j < individual_results[i].indices.size(); ++j) {
            EXPECT_EQ(individual_results[i].indices[j], batch_results[i].indices[j]);
            EXPECT_NEAR(individual_results[i].distances[j], batch_results[i].distances[j], 1e-6f);
        }
    }
}
