# Helix Architecture

## Overview

Helix is a production-grade vector similarity search library built in C++20 with GPU acceleration and Python bindings. The architecture is designed for high performance, scalability, and extensibility.

## Core Components

### IndexBase
- Abstract base class for all index implementations
- Defines common interface: `train()`, `add()`, `search()`, `save()`, `load()`
- Manages configuration, metadata, and ID generation
- Provides thread-safe operations

### Distance Metrics
- **L2Distance**: Euclidean distance with AVX2 SIMD optimization
- **InnerProductDistance**: Dot product similarity
- **CosineDistance**: Cosine similarity with normalization
- Fallback implementations for non-SIMD architectures

### Threading Infrastructure
- **ThreadPool**: Configurable thread pool for parallel operations
- **parallelFor**: Parallel iteration over vector ranges
- Automatic load balancing and work distribution

### Persistence Layer
- **Manifest**: Metadata storage with versioning and integrity checks
- **FileReader/FileWriter**: Atomic I/O operations with memory mapping
- Support for fvecs/ivecs dataset formats
- Cross-platform file operations

## Index Implementations

### IndexFlat
- **Algorithm**: Brute-force exact nearest neighbor search
- **Complexity**: O(n×d) search time, O(n×d) memory
- **Use Case**: Small datasets requiring exact results
- **Features**: SIMD-optimized distance computation, batch search

### IndexPQ (Product Quantization)
- **Algorithm**: Vector compression using sub-vector quantization
- **Complexity**: O(n×m) search time, O(n×m/8) memory
- **Use Case**: Memory-constrained applications
- **Features**: Asymmetric Distance Computation (ADC), configurable sub-vectors

### IndexIVF (Inverted File Index)
- **Algorithm**: K-means clustering with inverted file structure
- **Complexity**: O(nprobe×d) search time, O(n×d) memory
- **Use Case**: Large-scale approximate search
- **Features**: Configurable nprobe, cluster-based search

### IndexHNSW (Hierarchical Navigable Small World)
- **Algorithm**: Graph-based approximate nearest neighbor search
- **Complexity**: O(log n×d) search time, O(n×d×M) memory
- **Use Case**: High-dimensional, fast search
- **Features**: Multi-level graph navigation, configurable ef_search

## GPU Acceleration

### CUDA Backend
- **CudaIndexFlat**: GPU-accelerated brute-force search
- **Memory Management**: Efficient GPU memory allocation and deallocation
- **Device Management**: Multi-GPU support with device selection
- **Async Operations**: Non-blocking search operations

### CUDA Features
- Automatic GPU detection and fallback
- Memory pool for efficient allocation
- Batch processing for multiple queries
- Device synchronization and error handling

## Python Integration

### pybind11 Bindings
- **Zero-copy**: Direct NumPy array integration
- **Type Safety**: Automatic dimension validation
- **Context Management**: RAII-style resource management
- **API Consistency**: C++ and Python APIs match

### Python Features
- NumPy array compatibility
- Automatic memory management
- Exception handling and error messages
- Seamless integration with scientific Python stack

## Build System

### CMake Configuration
- **C++20 Standard**: Modern C++ features and optimizations
- **Optional Components**: Tests, benchmarks, Python bindings, CUDA
- **Cross-platform**: Linux, macOS, Windows support
- **Sanitizers**: AddressSanitizer, UndefinedBehaviorSanitizer

### Dependencies
- **Core**: STL, threading, filesystem
- **Python**: pybind11, NumPy
- **CUDA**: CUDA Toolkit 11.8+
- **Testing**: GoogleTest, GoogleBenchmark

## Performance Characteristics

| Index Type | Search Time | Memory Usage | Accuracy | Use Case |
|------------|-------------|--------------|----------|----------|
| IndexFlat | O(n×d) | O(n×d) | 100% | Exact search, small datasets |
| IndexPQ | O(n×m) | O(n×m/8) | 95-99% | Memory-constrained applications |
| IndexIVF | O(nprobe×d) | O(n×d) | 90-98% | Large-scale approximate search |
| IndexHNSW | O(log n×d) | O(n×d×M) | 95-99% | High-dimensional, fast search |

*Where n=vectors, d=dimensions, m=subvectors, nprobe=clusters searched*

## Testing Strategy

### Unit Tests
- **Coverage**: All public APIs and edge cases
- **Frameworks**: GoogleTest with custom matchers
- **Data**: Synthetic datasets with known properties
- **Validation**: Cross-platform compatibility

### Integration Tests
- **Persistence**: Round-trip save/load operations
- **Threading**: Concurrent access and race conditions
- **Memory**: Allocation patterns and leak detection
- **Performance**: Benchmark regression testing

### CI/CD Pipeline
- **Multi-compiler**: GCC, Clang, MSVC
- **Sanitizers**: Memory and undefined behavior detection
- **Cross-platform**: Linux, macOS, Windows
- **Python**: Multiple Python versions and NumPy compatibility

## Future Extensions

### Planned Features
- **Hybrid Indexing**: Sparse + dense vector support
- **Multimodal**: Text, image, audio embedding support
- **Distributed**: Multi-node cluster support
- **Encryption**: Searchable encryption for privacy

### Research Areas
- **SOAR Algorithm**: Next-generation ANN algorithms
- **Hardware Acceleration**: ARM Neon, AVX512 support
- **Federated Learning**: Privacy-preserving search
- **Edge Computing**: IoT and mobile deployment

## Design Principles

### Performance First
- SIMD optimizations for distance computation
- Memory-aligned allocations for cache efficiency
- Parallel processing for batch operations
- GPU acceleration for large-scale search

### Extensibility
- Plugin architecture for new index types
- Configurable distance metrics
- Modular build system
- Clean separation of concerns

### Reliability
- Comprehensive error handling
- Memory safety with RAII
- Thread-safe operations
- Extensive testing and validation

### Usability
- Consistent API across all index types
- Clear documentation and examples
- Python bindings for accessibility
- Cross-platform compatibility