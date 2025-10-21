#include "helix/cuda/cuda_simple.hpp"
#include "helix/benchmark/dataset.hpp"
#include "helix/benchmark/metrics.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <memory>

void benchmarkCudaFlat() {
    std::cout << "=== CUDA IndexFlat Benchmark ===" << std::endl;
    
    if (!helix::cuda_simple::isAvailable()) {
        std::cout << "CUDA not available, skipping CUDA benchmarks" << std::endl;
        return;
    }
    
    // Test parameters
    const helix::dim_t dimension = 128;
    const helix::idx_t numVectors = 10000;
    const helix::idx_t numQueries = 1000;
    const helix::idx_t k = 10;
    
    // Generate test data
    auto dataset = helix::BenchmarkDataset::generateSynthetic(numVectors, dimension, 0.0f, 1.0f);
    auto queries = helix::BenchmarkDataset::generateSynthetic(numQueries, dimension, 0.0f, 1.0f);
    
    // Create CUDA index
    helix::CudaIndexFlatSimple index(dimension, helix::MetricType::L2);
    
    // Train index
    auto start = std::chrono::high_resolution_clock::now();
    index.train(dataset.data.data(), dataset.nrows);
    auto train_time = std::chrono::high_resolution_clock::now() - start;
    
    // Add vectors
    start = std::chrono::high_resolution_clock::now();
    index.add(dataset.data.data(), dataset.nrows);
    auto add_time = std::chrono::high_resolution_clock::now() - start;
    
    // Benchmark search
    helix::MetricsCollector metrics;
    std::vector<helix::SearchResults> results;
    
    start = std::chrono::high_resolution_clock::now();
    for (helix::idx_t i = 0; i < numQueries; ++i) {
        auto query_start = std::chrono::high_resolution_clock::now();
        auto result = index.search(queries.data.data() + i * dimension, k);
        auto query_time = std::chrono::high_resolution_clock::now() - query_start;
        
        metrics.recordQuery(query_time);
        results.push_back(result);
    }
    auto total_search_time = std::chrono::high_resolution_clock::now() - start;
    
    // Print results
    std::cout << "Dataset: " << numVectors << " vectors, " << dimension << " dimensions" << std::endl;
    std::cout << "Queries: " << numQueries << " queries, k=" << k << std::endl;
    std::cout << "Train time: " << std::chrono::duration_cast<std::chrono::milliseconds>(train_time).count() << " ms" << std::endl;
    std::cout << "Add time: " << std::chrono::duration_cast<std::chrono::milliseconds>(add_time).count() << " ms" << std::endl;
    std::cout << "Total search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_search_time).count() << " ms" << std::endl;
    std::cout << "Average search time: " << metrics.getAverageLatency() << " ms" << std::endl;
    std::cout << "Throughput: " << metrics.getThroughput() << " queries/sec" << std::endl;
    std::cout << "GPU Memory Usage: " << "N/A" << " MB" << std::endl;
    
    // Benchmark batch search
    std::cout << "\n=== Batch Search Benchmark ===" << std::endl;
    
    const helix::idx_t batchSize = 100;
    std::vector<helix::SearchResults> batchResults;
    
    start = std::chrono::high_resolution_clock::now();
    for (helix::idx_t i = 0; i < numQueries; i += batchSize) {
        helix::idx_t currentBatchSize = std::min(batchSize, numQueries - i);
        std::vector<helix::SearchResults> currentResults;
        
        index.searchBatch(queries.data.data() + i * dimension, currentBatchSize, k, currentResults);
        batchResults.insert(batchResults.end(), currentResults.begin(), currentResults.end());
    }
    auto batch_search_time = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "Batch search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(batch_search_time).count() << " ms" << std::endl;
    std::cout << "Batch throughput: " << (numQueries * 1000.0) / std::chrono::duration_cast<std::chrono::milliseconds>(batch_search_time).count() << " queries/sec" << std::endl;
}

void benchmarkCudaPQ() {
    std::cout << "\n=== CUDA IndexPQ Benchmark ===" << std::endl;
    
    if (!helix::cuda_utils::isCudaAvailable()) {
        std::cout << "CUDA not available, skipping CUDA benchmarks" << std::endl;
        return;
    }
    
    // Test parameters
    const helix::dim_t dimension = 128;
    const helix::idx_t numVectors = 10000;
    const helix::idx_t numQueries = 1000;
    const helix::idx_t k = 10;
    
    // Generate test data
    auto dataset = helix::BenchmarkDataset::generateSynthetic(numVectors, dimension, 0.0f, 1.0f);
    auto queries = helix::BenchmarkDataset::generateSynthetic(numQueries, dimension, 0.0f, 1.0f);
    
    // Create CUDA index
    helix::IndexConfig config(dimension, helix::MetricType::L2, helix::IndexType::PQ);
    helix::CudaIndexPQ index(config);
    
    // Train index
    auto start = std::chrono::high_resolution_clock::now();
    index.train(dataset.data.data(), dataset.nrows);
    auto train_time = std::chrono::high_resolution_clock::now() - start;
    
    // Add vectors
    start = std::chrono::high_resolution_clock::now();
    index.add(dataset.data.data(), dataset.nrows);
    auto add_time = std::chrono::high_resolution_clock::now() - start;
    
    // Benchmark search
    helix::MetricsCollector metrics;
    std::vector<helix::SearchResults> results;
    
    start = std::chrono::high_resolution_clock::now();
    for (helix::idx_t i = 0; i < numQueries; ++i) {
        auto query_start = std::chrono::high_resolution_clock::now();
        auto result = index.search(queries.data.data() + i * dimension, k);
        auto query_time = std::chrono::high_resolution_clock::now() - query_start;
        
        metrics.recordQuery(query_time);
        results.push_back(result);
    }
    auto total_search_time = std::chrono::high_resolution_clock::now() - start;
    
    // Print results
    std::cout << "Dataset: " << numVectors << " vectors, " << dimension << " dimensions" << std::endl;
    std::cout << "Queries: " << numQueries << " queries, k=" << k << std::endl;
    std::cout << "Train time: " << std::chrono::duration_cast<std::chrono::milliseconds>(train_time).count() << " ms" << std::endl;
    std::cout << "Add time: " << std::chrono::duration_cast<std::chrono::milliseconds>(add_time).count() << " ms" << std::endl;
    std::cout << "Total search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_search_time).count() << " ms" << std::endl;
    std::cout << "Average search time: " << metrics.getAverageLatency() << " ms" << std::endl;
    std::cout << "Throughput: " << metrics.getThroughput() << " queries/sec" << std::endl;
    std::cout << "GPU Memory Usage: " << "N/A" << " MB" << std::endl;
    
    // Benchmark batch search
    std::cout << "\n=== Batch Search Benchmark ===" << std::endl;
    
    const helix::idx_t batchSize = 100;
    std::vector<helix::SearchResults> batchResults;
    
    start = std::chrono::high_resolution_clock::now();
    for (helix::idx_t i = 0; i < numQueries; i += batchSize) {
        helix::idx_t currentBatchSize = std::min(batchSize, numQueries - i);
        std::vector<helix::SearchResults> currentResults;
        
        index.searchBatch(queries.data.data() + i * dimension, currentBatchSize, k, currentResults);
        batchResults.insert(batchResults.end(), currentResults.begin(), currentResults.end());
    }
    auto batch_search_time = std::chrono::high_resolution_clock::now() - start;
    
    std::cout << "Batch search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(batch_search_time).count() << " ms" << std::endl;
    std::cout << "Batch throughput: " << (numQueries * 1000.0) / std::chrono::duration_cast<std::chrono::milliseconds>(batch_search_time).count() << " queries/sec" << std::endl;
}

void benchmarkCpuGpuComparison() {
    std::cout << "\n=== CPU vs GPU Performance Comparison ===" << std::endl;
    
    if (!helix::cuda_utils::isCudaAvailable()) {
        std::cout << "CUDA not available, skipping comparison" << std::endl;
        return;
    }
    
    // Test parameters
    const helix::dim_t dimension = 128;
    const helix::idx_t numVectors = 5000;
    const helix::idx_t numQueries = 500;
    const helix::idx_t k = 10;
    
    // Generate test data
    auto dataset = helix::BenchmarkDataset::generateSynthetic(numVectors, dimension, 0.0f, 1.0f);
    auto queries = helix::BenchmarkDataset::generateSynthetic(numQueries, dimension, 0.0f, 1.0f);
    
    // CPU IndexFlat
    helix::IndexConfig config(dimension, helix::MetricType::L2, helix::IndexType::Flat);
    helix::IndexFlat cpu_index(config);
    
    auto start = std::chrono::high_resolution_clock::now();
    cpu_index.train(dataset.data.data(), dataset.nrows);
    cpu_index.add(dataset.data.data(), dataset.nrows);
    auto cpu_setup_time = std::chrono::high_resolution_clock::now() - start;
    
    start = std::chrono::high_resolution_clock::now();
    for (helix::idx_t i = 0; i < numQueries; ++i) {
        cpu_index.search(queries.data.data() + i * dimension, k);
    }
    auto cpu_search_time = std::chrono::high_resolution_clock::now() - start;
    
    // GPU IndexFlat
    helix::CudaIndexFlat gpu_index(config);
    
    start = std::chrono::high_resolution_clock::now();
    gpu_index.train(dataset.data.data(), dataset.nrows);
    gpu_index.add(dataset.data.data(), dataset.nrows);
    auto gpu_setup_time = std::chrono::high_resolution_clock::now() - start;
    
    start = std::chrono::high_resolution_clock::now();
    for (helix::idx_t i = 0; i < numQueries; ++i) {
        gpu_index.search(queries.data.data() + i * dimension, k);
    }
    auto gpu_search_time = std::chrono::high_resolution_clock::now() - start;
    
    // Print comparison
    std::cout << "Dataset: " << numVectors << " vectors, " << dimension << " dimensions" << std::endl;
    std::cout << "Queries: " << numQueries << " queries, k=" << k << std::endl;
    std::cout << "\nCPU Performance:" << std::endl;
    std::cout << "  Setup time: " << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_setup_time).count() << " ms" << std::endl;
    std::cout << "  Search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_search_time).count() << " ms" << std::endl;
    std::cout << "  Throughput: " << (numQueries * 1000.0) / std::chrono::duration_cast<std::chrono::milliseconds>(cpu_search_time).count() << " queries/sec" << std::endl;
    
    std::cout << "\nGPU Performance:" << std::endl;
    std::cout << "  Setup time: " << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_setup_time).count() << " ms" << std::endl;
    std::cout << "  Search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(gpu_search_time).count() << " ms" << std::endl;
    std::cout << "  Throughput: " << (numQueries * 1000.0) / std::chrono::duration_cast<std::chrono::milliseconds>(gpu_search_time).count() << " queries/sec" << std::endl;
    std::cout << "  Memory usage: " << gpu_index.getGpuMemoryUsage() / (1024 * 1024) << " MB" << std::endl;
    
    // Calculate speedup
    double speedup = static_cast<double>(cpu_search_time.count()) / gpu_search_time.count();
    std::cout << "\nGPU Speedup: " << speedup << "x" << std::endl;
}

void benchmarkMemoryScaling() {
    std::cout << "\n=== Memory Scaling Benchmark ===" << std::endl;
    
    if (!helix::cuda_utils::isCudaAvailable()) {
        std::cout << "CUDA not available, skipping memory scaling" << std::endl;
        return;
    }
    
    const helix::dim_t dimension = 128;
    const helix::idx_t k = 10;
    
    std::vector<helix::idx_t> datasetSizes = {1000, 5000, 10000, 20000};
    
    for (helix::idx_t numVectors : datasetSizes) {
        std::cout << "\nTesting with " << numVectors << " vectors:" << std::endl;
        
        try {
            // Generate test data
            auto dataset = helix::BenchmarkDataset::generateSynthetic(numVectors, dimension, 0.0f, 1.0f);
            auto queries = helix::BenchmarkDataset::generateSynthetic(100, dimension, 0.0f, 1.0f);
            
            // Create CUDA index
            helix::IndexConfig config(dimension, helix::MetricType::L2, helix::IndexType::Flat);
            helix::CudaIndexFlat index(config);
            
            // Train and add
            auto start = std::chrono::high_resolution_clock::now();
            index.train(dataset.data.data(), dataset.nrows);
            index.add(dataset.data.data(), dataset.nrows);
            auto setup_time = std::chrono::high_resolution_clock::now() - start;
            
            // Search
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 100; ++i) {
                index.search(queries.data.data() + i * dimension, k);
            }
            auto search_time = std::chrono::high_resolution_clock::now() - start;
            
            std::cout << "  Setup time: " << std::chrono::duration_cast<std::chrono::milliseconds>(setup_time).count() << " ms" << std::endl;
            std::cout << "  Search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(search_time).count() << " ms" << std::endl;
            std::cout << "  Memory usage: " << index.getGpuMemoryUsage() / (1024 * 1024) << " MB" << std::endl;
            std::cout << "  Throughput: " << (100 * 1000.0) / std::chrono::duration_cast<std::chrono::milliseconds>(search_time).count() << " queries/sec" << std::endl;
            
        } catch (const helix::HelixException& e) {
            std::cout << "  Failed: " << e.what() << std::endl;
        }
    }
}

int main() {
    std::cout << "Helix CUDA Benchmark Suite" << std::endl;
    std::cout << "===========================" << std::endl;
    
    // Print CUDA device info
    if (helix::cuda_utils::isCudaAvailable()) {
        helix::cuda_utils::printDeviceInfo();
        std::cout << std::endl;
    }
    
    // Run benchmarks
    benchmarkCudaFlat();
    benchmarkCudaPQ();
    benchmarkCpuGpuComparison();
    benchmarkMemoryScaling();
    
    std::cout << "\nBenchmark completed!" << std::endl;
    return 0;
}
