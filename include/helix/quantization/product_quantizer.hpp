#pragma once

#include"helix/common/types.hpp"
#include"helix/core/distance.hpp"
#include<vector>
#include<memory>

namespace helix {

class ProductQuantizer {
public:
    ProductQuantizer(dim_t dimension,int nsub,int nbits);
    
    void train(const float* vectors,idx_t n);
    
    void encode(const float* vectors,idx_t n,uint8_t* codes) const;
    void decode(const uint8_t* codes,idx_t n,float* vectors) const;
    
    void computeDistanceTable(const float* query,float* distTable) const;
    float computeAsymmetricDistance(const float* query,const uint8_t* code) const;
    
    dim_t dimension() const { return dim_; }
    int nsub() const { return nsub_; }
    int nbits() const { return nbits_; }
    int ksub() const { return ksub_; }
    int dsub() const { return dsub_; }
    
    bool isTrained() const { return isTrained_; }
    
    const float* centroids() const { return centroids_.data(); }
    
    void loadCentroids(const float* centroids,size_t size);
    void markTrained() { isTrained_=true; }
    
private:
    void trainSubspace(const float* vectors,idx_t n,int subIdx);
    void kmeansSubspace(const float* vectors,idx_t n,int subIdx);
    
    dim_t dim_;
    int nsub_;
    int nbits_;
    int ksub_;
    int dsub_;
    bool isTrained_;
    
    std::vector<float> centroids_;
};

}

