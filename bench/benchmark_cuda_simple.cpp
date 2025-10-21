#include "helix/cuda/cuda_simple.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

void benchmarkCudaSimple() {
    std::cout << "=== CUDA Simple Benchmark ===" << std::endl;
    
    if (!helix::cuda_simple::isAvailable()) {
        std::cout << "CUDA not available, skipping CUDA benchmarks" << std::endl;
        return;
    }
    
    // Test parameters
    const helix::dim_t dimension = 128;
    const helix::idx_t numVectors = 1000;
    const helix::idx_t numQueries = 100;
    const helix::idx_t k = 10;
    
    // Generate test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    std::vector<float> dataset(numVectors * dimension);
    std::vector<float> queries(numQueries * dimension);
    
    for (size_t i = 0; i < dataset.size(); ++i) {
        dataset[i] = dis(gen);
    }
    
    for (size_t i = 0; i < queries.size(); ++i) {
        queries[i] = dis(gen);
    }
    
    // Create CUDA index
    helix::CudaIndexFlatSimple index(dimension, helix::MetricType::L2);
    
    // Train index
    auto start = std::chrono::high_resolution_clock::now();
    index.train(dataset.data(), numVectors);
    auto train_time = std::chrono::high_resolution_clock::now() - start;
    
    // Add vectors
    start = std::chrono::high_resolution_clock::now();
    index.add(dataset.data(), numVectors);
    auto add_time = std::chrono::high_resolution_clock::now() - start;
    
    // Benchmark search
    start = std::chrono::high_resolution_clock::now();
    for (helix::idx_t i = 0; i < numQueries; ++i) {
        auto result = index.search(queries.data() + i * dimension, k);
    }
    auto total_search_time = std::chrono::high_resolution_clock::now() - start;
    
    // Print results
    std::cout << "Dataset: " << numVectors << " vectors, " << dimension << " dimensions" << std::endl;
    std::cout << "Queries: " << numQueries << " queries, k=" << k << std::endl;
    std::cout << "Train time: " << std::chrono::duration_cast<std::chrono::milliseconds>(train_time).count() << " ms" << std::endl;
    std::cout << "Add time: " << std::chrono::duration_cast<std::chrono::milliseconds>(add_time).count() << " ms" << std::endl;
    std::cout << "Total search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(total_search_time).count() << " ms" << std::endl;
    std::cout << "Average search time: " << std::chrono::duration_cast<std::chrono::microseconds>(total_search_time).count() / numQueries << " μs" << std::endl;
    std::cout << "Throughput: " << (numQueries * 1000.0) / std::chrono::duration_cast<std::chrono::milliseconds>(total_search_time).count() << " queries/sec" << std::endl;
    
    // Test single search
    auto single_start = std::chrono::high_resolution_clock::now();
    auto result = index.search(queries.data(), k);
    auto single_time = std::chrono::high_resolution_clock::now() - single_start;
    
    std::cout << "Single search time: " << std::chrono::duration_cast<std::chrono::microseconds>(single_time).count() << " μs" << std::endl;
    std::cout << "Results size: " << result.results.size() << std::endl;
}

void benchmarkMemoryScaling() {
    std::cout << "\n=== Memory Scaling Benchmark ===" << std::endl;
    
    if (!helix::cuda_simple::isAvailable()) {
        std::cout << "CUDA not available, skipping memory scaling" << std::endl;
        return;
    }
    
    const helix::dim_t dimension = 128;
    const helix::idx_t k = 10;
    
    std::vector<helix::idx_t> datasetSizes = {100, 500, 1000, 2000};
    
    for (helix::idx_t numVectors : datasetSizes) {
        std::cout << "\nTesting with " << numVectors << " vectors:" << std::endl;
        
        try {
            // Generate test data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.0f, 1.0f);
            
            std::vector<float> dataset(numVectors * dimension);
            std::vector<float> query(dimension);
            
            for (size_t i = 0; i < dataset.size(); ++i) {
                dataset[i] = dis(gen);
            }
            
            for (size_t i = 0; i < query.size(); ++i) {
                query[i] = dis(gen);
            }
            
            // Create CUDA index
            helix::CudaIndexFlatSimple index(dimension, helix::MetricType::L2);
            
            // Train and add
            auto start = std::chrono::high_resolution_clock::now();
            index.train(dataset.data(), numVectors);
            index.add(dataset.data(), numVectors);
            auto setup_time = std::chrono::high_resolution_clock::now() - start;
            
            // Search
            start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 10; ++i) {
                index.search(query.data(), k);
            }
            auto search_time = std::chrono::high_resolution_clock::now() - start;
            
            std::cout << "  Setup time: " << std::chrono::duration_cast<std::chrono::milliseconds>(setup_time).count() << " ms" << std::endl;
            std::cout << "  Search time: " << std::chrono::duration_cast<std::chrono::milliseconds>(search_time).count() << " ms" << std::endl;
            std::cout << "  Throughput: " << (10 * 1000.0) / std::chrono::duration_cast<std::chrono::milliseconds>(search_time).count() << " queries/sec" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "  Failed: " << e.what() << std::endl;
        }
    }
}

int main() {
    std::cout << "Helix CUDA Simple Benchmark Suite" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Print CUDA device info
    if (helix::cuda_simple::isAvailable()) {
        int deviceCount = helix::cuda_simple::getDeviceCount();
        std::cout << "CUDA device count: " << deviceCount << std::endl;
        std::cout << "Free GPU memory: " << helix::cuda_simple::getFreeMemory() / (1024 * 1024) << " MB" << std::endl;
        std::cout << std::endl;
    }
    
    // Run benchmarks
    benchmarkCudaSimple();
    benchmarkMemoryScaling();
    
    std::cout << "\nBenchmark completed!" << std::endl;
    return 0;
}
