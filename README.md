# Helix

Vector similarity search library in C++20.

## Implemented

- IndexFlat (exact brute-force search)
- IndexPQ (product quantization with ADC)
- L2, inner product, and cosine distance metrics
- Parallel batch search
- Index persistence (save/load)
- Dataset loaders (fvecs/ivecs, synthetic generation)
- Benchmarking infrastructure (recall, latency, throughput metrics)
- Product quantization with k-means codebook training

## In Progress

- IndexIVF implementation

## Architecture

```mermaid
graph TD
    A[User Application] --> B[IndexFlat]
    B --> C[IndexBase Interface]
    C --> D[DistanceComputer]
    D --> E[L2Distance]
    D --> F[InnerProductDistance]
    D --> G[CosineDistance]
    B --> H[ThreadPool]
    B --> I[FileReader/Writer]
    B --> J[Vector Storage]
    H --> K[parallelFor]
    E --> L[SIMD Operations]
    F --> L
```

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

## License

MIT License
