#pragma once

#include"helix/common/types.hpp"
#include<vector>
#include<random>

namespace helix {

class KMeans
{
  public:
    KMeans(int k,int maxIter=100);
    void train(const float* data,idx_t n,dim_t dim);
    idx_t assign(const float* vector,dim_t dim) const;
    const std::vector<float>& getCentroids() const { return centroids_; }
    bool isTrained() const { return trained_; }
    void setCentroids(const std::vector<float>& centroids);
    void reset();

  private:
    int k_;
    int maxIter_;
    dim_t dim_;
    std::vector<float> centroids_;
    bool trained_;
    mutable std::mt19937 rng_;

    void initializeCentroids(const float* data,idx_t n);
    void updateCentroids(const float* data,idx_t n,const std::vector<idx_t>& assignments);
    float computeDistance(const float* a,const float* b) const;
};

} // namespace helix
