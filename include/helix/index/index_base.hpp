#pragma once

#include "helix/common/types.hpp"
#include "helix/common/utils.hpp"
#include <string>
#include <vector>

namespace helix {

class IndexBase {
  public:
    explicit IndexBase(const IndexConfig &config) : config_(config), ntotal_(0) {}
    virtual ~IndexBase()= default;

    virtual void train(const float *vectors, idx_t n)= 0;
    virtual void add(const float *vectors, idx_t n)= 0;
    virtual void addWithIds(const float *vectors, const idx_t *ids, idx_t n)= 0;
    virtual SearchResults search(const float *query, int k) const= 0;
    virtual void searchBatch(const float *queries, idx_t nq, int k,
                             std::vector<SearchResults> &results) const= 0;

    virtual void save(const std::string &path) const= 0;
    virtual void load(const std::string &path)= 0;

    idx_t ntotal() const { return ntotal_; }
    dim_t dimension() const { return config_.dimension; }
    MetricType metric() const { return config_.metric; }
    bool isTrained() const { return isTrained_; }

  protected:
    IndexConfig config_;
    idx_t ntotal_;
    bool isTrained_= false;
};

} // namespace helix
