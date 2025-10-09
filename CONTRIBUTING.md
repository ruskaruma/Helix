# Contributing to Helix

Thank you for your interest in contributing to Helix. This document provides guidelines for contributions.

## Development Setup

1. Clone the repository
2. Install dependencies (CMake 3.20+, C++20 compiler)
3. Build the project:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug -DHELIX_ENABLE_SANITIZERS=ON
   cmake --build . -j
   ```

## Testing

All new code should include unit tests. Run tests before submitting:
```bash
cd build
ctest --output-on-failure
```

## Pull Request Process

1. Create a feature branch from main
2. Make your changes with clear commit messages
3. Ensure all tests pass
4. Update documentation if needed
5. Submit PR with description of changes

## Issue Labels

- `good-first-issue` - Good for newcomers
- `performance` - Performance optimization
- `docs` - Documentation improvements
- `bug` - Bug fixes
- `design` - Architecture and design discussions

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

