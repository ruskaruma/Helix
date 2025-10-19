#pragma once

#include "helix/core/distance.hpp"
#include "helix/core/vector_utils.hpp"
#include "helix/index/index_base.hpp"
#include <algorithm>
#include <vector>

namespace helix {

class IndexFlat : public IndexBase
{
  public:
    explicit IndexFlat(const IndexConfig &config);
    ~IndexFlat() override= default;
    void train(const float *vectors, idx_t n) override;
    void add(const float *vectors, idx_t n) override;
    void addWithIds(const float *vectors, const idx_t *ids, idx_t n) override;
    SearchResults search(const float *query, int k) const override;
    void searchBatch(const float *queries, idx_t nq, int k,
                     std::vector<SearchResults> &results) const override;

    void save(const std::string &path) const override;
    void load(const std::string &path) override;

    const float *getVectors() const { return vectors_.data(); }
    const idx_t *getIds() const { return ids_.data(); }

  private:
    std::vector<float> vectors_;
    std::vector<idx_t> ids_;
    std::unique_ptr<DistanceComputer> distance_;
};

} // namespace helix
