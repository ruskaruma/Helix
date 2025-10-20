# Architecture

minimal, factual overview of current components.

## modules
- core: distance, memory, io, vector_utils, threading
- indexes: IndexFlat, IndexPQ, IndexIVF, IndexHNSW
- quantization: ProductQuantizer (k-means, ADC)
- bench: dataset loaders, metrics
- tests: unit tests

## diagram

```mermaid
graph LR
    A[Application] --> B[Index API]
    B --> C[IndexFlat]
    B --> D[IndexPQ]
    B --> E[IndexIVF]
    B --> F[IndexHNSW]

    subgraph Core
        G[Distance]
        H[ThreadPool]
        I[IO]
        J[Vector Utils]
    end

    C --> G
    D --> G
    E --> G
    F --> G

    C --> H
    D --> H
    E --> H
    F --> H

    C --> I
    D --> I
    E --> I
    F --> I
```
