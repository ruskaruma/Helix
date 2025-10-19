#pragma once

#include "helix/common/types.hpp"
#include <cmath>
#include <immintrin.h>

namespace helix {

class DistanceComputer {
  public:
    virtual ~DistanceComputer()= default;
    virtual float compute(const float *a, const float *b, dim_t dim) const= 0;
    virtual void computeBatch(const float *queries, const float *database, idx_t nq, idx_t nb,
                              dim_t dim, float *distances) const= 0;
};

class L2Distance : public DistanceComputer {
  public:
    float compute(const float *a, const float *b, dim_t dim) const override;
    void computeBatch(const float *queries, const float *database, idx_t nq, idx_t nb, dim_t dim,
                      float *distances) const override;

  private:
    float computeSimd(const float *a, const float *b, dim_t dim) const;
};

class InnerProductDistance : public DistanceComputer {
  public:
    float compute(const float *a, const float *b, dim_t dim) const override;
    void computeBatch(const float *queries, const float *database, idx_t nq, idx_t nb, dim_t dim,
                      float *distances) const override;

  private:
    float computeSimd(const float *a, const float *b, dim_t dim) const;
};

class CosineDistance : public DistanceComputer {
  public:
    float compute(const float *a, const float *b, dim_t dim) const override;
    void computeBatch(const float *queries, const float *database, idx_t nq, idx_t nb, dim_t dim,
                      float *distances) const override;
};

std::unique_ptr<DistanceComputer> createDistanceComputer(MetricType metric);

} // namespace helix
