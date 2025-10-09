# Helix

A research-first, production-grade semantic search engine and vector similarity library implemented in modern C++ (C++20) with GPU acceleration, Python bindings, and deployable server components.

## Overview

Helix is a high-performance vector indexer and search engine designed for production use. It provides FAISS-like functionality with a focus on reproducibility, benchmarking, and systems engineering best practices.

## Key Features

- **Performance**: Low-latency single-query p50 < 2ms for small indexes, GPU batched throughput > 50k qps on 1M vectors
- **Recall**: Configurable ANN with recall@10 >= 0.95 on standard benchmarks
- **Scalability**: Sharding and partitioning for >100M vectors with streaming ingest
- **Engineering**: Robust CI, 80% unit coverage on core, reproducible benchmarks
- **Usability**: Python SDK, CLI tools, and gRPC server for distributed queries

## Planned Index Types

- IndexFlatL2 (exact baseline)
- IndexIVF (inverted file with kmeans coarse quantizer)
- IndexPQ (product quantization storage)
- IndexIVF+PQ (per-cluster PQ)
- IndexHNSW (graph-based ANN)
- IndexHybrid (coarse filter + HNSW rerank)

## Build Requirements

- C++20 compatible compiler (GCC 12+ or Clang 15+)
- CMake 3.20+
- CUDA Toolkit 11.8+ (optional, for GPU acceleration)
- Python 3.10+ (optional, for Python bindings)

## Building from Source

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest --output-on-failure
```

### Build Options

- `HELIX_BUILD_TESTS` - Build unit tests (default: ON)
- `HELIX_BUILD_BENCH` - Build benchmarks (default: ON)
- `HELIX_BUILD_PYTHON` - Build Python bindings (default: ON)
- `HELIX_BUILD_CUDA` - Build CUDA kernels (default: OFF)
- `HELIX_ENABLE_SANITIZERS` - Enable address/undefined sanitizers (default: OFF)

## Project Status

This project is under active development. Current milestone: Core infrastructure and IndexFlatL2 implementation.

## License

MIT License - see LICENSE file for details

## Roadmap

### 90 Days
- C++20 skeleton repo
- IndexFlatL2 implementation
- pybind11 bindings for FlatL2
- Small benchmark harness
- CI + README

### 6 Months
- IVF + PQ + HNSW CPU implementations
- Persistence and mmapped IO
- Recall vs latency benchmarks
- Packaged wheels

### 12 Months
- CUDA kernels for PQ ADC and brute-force
- Distributed query prototype
- Production-grade docs and k8s manifests
