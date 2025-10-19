#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace helix {
using idx_t= std::int64_t;
using dim_t= std::int32_t;
struct SearchResult {
    idx_t id;
    float distance;

    SearchResult()= default;
    SearchResult(idx_t id_, float distance_) : id(id_), distance(distance_) {}

    bool operator<(const SearchResult &other) const { return distance < other.distance; }
};
struct SearchResults {
    std::vector<SearchResult> results;
    SearchResults()= default;
    explicit SearchResults(size_t k) { results.reserve(k); }
};
enum class MetricType { L2, InnerProduct, Cosine };

enum class IndexType { Flat, IVF, PQ, IVFPQ, HNSW, Hybrid };
struct IndexConfig {
    dim_t dimension;
    MetricType metric;
    IndexType type;

    IndexConfig(dim_t dim, MetricType met, IndexType typ)
        : dimension(dim), metric(met), type(typ) {}
};
} // namespace helix
